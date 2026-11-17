#!/usr/bin/env python3
"""評価用ステムの集まり（`data/stems/<曲ID>/`）に対して、M0 の3指標をまとめて出す（回帰テスト）。

使い方::

    python run_eval.py data/stems [--out data/results/YYYY-MM-DD.md] [--est-dir 推定の置き場] [--max-seconds 60]
                       [--model analysis/models/crnn_synth_v0.onnx] [--render] [--kit-size 12] [--choke all|hihat|none] [--only a01,a07]

各曲のフォルダ:
    drums.wav | .flac    分離済みのドラムのステム（必須。フォルダ内の音声ファイルが 1 つなら名前は問わない）
    hits.csv | hits.mid  打点の正解（あれば M0-1 を出す）
    キック.wav スネア.wav ハイハット.wav （シンバル.wav）
                         分離ツールが出すパート別ステム。正解が無いときは、これの打点検出を「疑似正解」にして F 値の目安を出す
                         （song フォルダに hits_pseudo.csv を書く）
フォルダを作らず `data/stems/a01.wav` のように音声ファイルを直接並べてもよい（曲ID はファイル名）。

採譜の推定は、`--est-dir` に `<曲ID>.csv` があればそれを使い、無ければ基準値（`baseline_transcribe`）で作る。
出力は Markdown の表（曲ごとの F 値、切り出せた打の数と層、再現との STFT 距離）。
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drumeval import Hit, band_balance_db, load_hits, multi_resolution_stft_distance, part_stft_distance, transcription_scores  # noqa: E402
from drumeval.baseline import onsets_from_part, transcribe  # noqa: E402
from drumeval.extract import (  # noqa: E402
    attach_confidence, build_kit, build_kit_by_features, build_kit_by_level, cut_segments, cut_segments_from_parts, estimate_velocities, head_embedding,
    hit_segments_from_parts,
    SAMPLE_LEAD_SEC, _onset_lag, level_for_velocity, level_range_for, refine_onsets, render_kit, render_midi, segment_features,
    self_render_from_parts, silent_hits, velocity_for_level,
)
from drumeval.labels import M0_CLASSES, save_hits_midi as write_hits_midi, to_coarse  # noqa: E402
from stft_distance import load_mono  # noqa: E402


from drumeval.stems import (  # noqa: E402,F401  (find_stem などは drumeval.stems に移した。ここからも使える)
    AUDIO_EXT, PART_NAMES, detect_drum_start, find_part_stems, find_stem, read_label_offset, read_start_override, resolve_start,
    truth_hits_in_window,
)


def pseudo_truth_from_parts(parts: dict[str, Path], max_seconds: float, sr_target: int | None, start_sec: float = 0.0,
                            levels_out: dict[str, np.ndarray] | None = None) -> list[Hit]:
    """パート別ステムの打点検出を「疑似的な正解」にする。分離の誤りはそのまま残るので目安。時刻は start_sec からの相対。

    `levels_out` を渡すと、種類ごとの打の大きさ（大きい打を 0 dB とした相対値）を入れる（小さい打が多いステムは漏れを疑う）。
    """
    hits: list[Hit] = []
    for cls, path in parts.items():
        x, sr = load_mono(str(path), sr_target)
        x = x[int(start_sec * sr) :]
        if max_seconds and x.size > max_seconds * sr:
            x = x[: int(max_seconds * sr)]
        times, levels = onsets_from_part(x, sr, cls=cls, return_levels=True)
        if levels_out is not None:
            levels_out[cls] = levels
        for t in times:
            hits.append(Hit(float(t), cls, None))
    hits.sort(key=lambda h: h.time)
    return hits


def list_songs(stems_dir: Path) -> list[tuple[str, Path, Path]]:
    """(曲ID, フォルダ, ステムのファイル) の一覧。フォルダごとでも、音声ファイルが直接並んでいてもよい。"""
    songs = []
    for p in sorted(stems_dir.iterdir()):
        if p.is_dir():
            f = find_stem(p)
            if f:
                songs.append((p.name, p, f))
        elif p.suffix.lower() in AUDIO_EXT:
            songs.append((p.stem, stems_dir, p))
    return songs


def timeline(x: np.ndarray, renders: dict[str, np.ndarray], hits: list[Hit], sr: int, win_sec: float = 2.0, hop_sec: float = 1.0) -> list[dict]:
    """2 秒の窓ごとの距離（各再現 vs 元）と、窓の中の種類別の打数。どの区間が悪いかを見る。"""
    from collections import Counter

    n_win = max(int((x.size / sr - win_sec) / hop_sec) + 1, 1)
    rows = []
    for k in range(n_win):
        t0 = k * hop_sec
        a, b = int(t0 * sr), int((t0 + win_sec) * sr)
        if b - a < int(0.5 * sr):
            break
        row = {"start": t0, "end": t0 + win_sec}
        for name, y in renders.items():
            row[name] = multi_resolution_stft_distance(x[a:b], y[a:b])["total"]
        if "self" in renders:
            e_x = float(np.sum(x[a:b] ** 2)) + 1e-12
            e_res = float(np.sum((x[a:b] - renders["self"][a:b]) ** 2)) + 1e-12
            row["unexplained_db"] = 10.0 * np.log10(e_res / e_x)  # 0 dB に近いほど、打点で説明できない音が多い
        c = Counter(to_coarse(h.cls) for h in hits if t0 <= h.time < t0 + win_sec)
        for cls in ("kick", "snare", "hihat", "tom", "cymbal"):
            row[cls] = c.get(cls, 0)
        rows.append(row)
    return rows


def write_audition(kit, path: Path, max_len_sec: float = 0.6, gap_sec: float = 0.25) -> None:
    """切り出した音を種類 → 層（弱→強）→ ラウンドロビンの順に並べた試聴用 WAV。切り出しそのものの質を耳で確かめる。"""
    import soundfile as sf

    sr = kit.sr
    pieces = []
    for cls in ("kick", "snare", "hihat", "tom", "cymbal"):
        for layer in kit.parts.get(cls, []):
            for smp in layer.samples:
                pieces.append(smp[: int(max_len_sec * sr)])
                pieces.append(np.zeros(int(gap_sec * sr)))
        if cls in kit.parts:
            pieces.append(np.zeros(int(0.6 * sr)))
    if not pieces:
        return
    y = np.concatenate(pieces)
    sf.write(path, (y / (np.max(np.abs(y)) + 1e-9) * 0.9).astype(np.float32), sr)


def _match_rms(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """y の RMS を x に合わせる（全体の音量差で STFT 距離が膨らまないようにする）。"""
    rx = np.sqrt(np.mean(x**2)) + 1e-9
    ry = np.sqrt(np.mean(y**2)) + 1e-9
    return y * (rx / ry)


def _fmt_bands(d: dict[str, float]) -> str:
    """帯域ごとの差（dB）を 1 つの欄に: 「低+0.1 中低-0.2 中-1.0 高-4.2 超高-6.5」。"""
    return " ".join(f"{k}{v:+.1f}" for k, v in d.items())


def _counts(hits: list[Hit]) -> str:
    from collections import Counter

    c = Counter(to_coarse(h.cls) for h in hits)
    return " ".join(f"{k[:2]}{c[k]}" for k in ("kick", "snare", "hihat", "tom", "cymbal") if c.get(k))


CHOKE_POLICIES: dict[str, bool | set[str]] = {"all": True, "hihat": {"hihat"}, "none": False}


def parse_kit_size(text: str) -> dict[str, int]:
    """'24' か 'kick=48,snare=24,hihat=24,cymbal=24,tom=24' を 種類 → 数 にする（書いていない種類は既定）。'all' なら全部。"""
    out = {c: 24 for c in ("kick", "snare", "hihat", "tom", "cymbal")}
    if text.strip().lower() == "all":
        return {c: 0 for c in out}
    if "=" not in text:
        return {c: int(text) for c in out}
    for pair in text.split(","):
        if pair.strip():
            k, v = pair.split("=")
            out[k.strip()] = int(v)
    return out


def evaluate_song(song_dir: Path, est_csv: Path | None, max_seconds: float, song_id: str | None = None, wav: Path | None = None,
                  model: str | None = None, render: bool = False, threshold: float | None = None, kit_size: str = "24",
                  choke: str = "all", start_sec: float | None = None, fit_fx: bool = False, fit_seconds: float = 20.0,
                  parts_mode: str = "sl", save_kit_dir: str | None = None, nmfd_residual: bool = True, nmfd_options: dict | None = None,
                  nmfd_legacy: bool = False, tag: str = "") -> dict:
    wav = wav or find_stem(song_dir)

    def out_path(name: str) -> Path:
        """書き出すファイルの置き場。`tag` があれば名前の末尾に付ける（rerender_parts_v5.wav）。モデルを替えた結果を並べて聞ける。"""
        base, ext = name.rsplit(".", 1)
        return song_dir / (f"{base}_{tag}.{ext}" if tag else name)

    x_full, sr = load_mono(str(wav), None)
    # 評価の開始位置: start.txt → --start → ドラムが鳴り始める位置の自動検出（キックとスネアのパート別ステムがあればその合計で）
    start = resolve_start(song_dir, x_full, sr, start_sec, load=lambda path, target_sr: load_mono(str(path), target_sr)[0])
    x = x_full[int(start * sr) :]
    if max_seconds and x.size > max_seconds * sr:
        x = x[: int(max_seconds * sr)]
    row: dict = {"id": song_id or song_dir.name, "start": start, "seconds": x.size / sr}

    if est_csv and est_csv.exists():
        est = load_hits(est_csv)
        row["est_source"] = est_csv.name
    elif model:
        from drumeval.infer import transcribe_onnx

        est = transcribe_onnx(x, sr, model, threshold)
        row["est_source"] = Path(model).stem + ("" if threshold is None else f"@{threshold:g}")  # 既定は json の種類別しきい値
    else:
        est = transcribe(x, sr)
        row["est_source"] = "baseline"
    est = [h for h in est if h.time < x.size / sr]
    row["est_counts"] = _counts(est)

    truth_path = next((p for p in (song_dir / "hits.csv", song_dir / "hits.mid", song_dir / "hits.midi") if p.exists()), None)
    parts = find_part_stems(song_dir)
    ref: list[Hit] | None = None
    if truth_path:
        # 正解の時刻はファイルの先頭から。評価の区間（start から）の相対時刻に直す。label_offset.txt があれば、正解と音のずれも直す
        label_off = read_label_offset(song_dir)
        ref = truth_hits_in_window(load_hits(truth_path), start, x.size / sr, label_off)
        row["truth"] = "正解" + (f"（時刻を {label_off * 1000:+.0f} ms 直した）" if abs(label_off) >= 0.0005 else "")
    elif all(c in parts for c in M0_CLASSES):
        levels: dict[str, np.ndarray] = {}
        ref = pseudo_truth_from_parts(parts, x.size / sr, sr, start, levels_out=levels)  # シンバル・タムのステムがあればそれも含める
        row["truth"] = "疑似（パート別ステム）"
        # 種類ごとの打の大きさの分布: 0〜−10 ／ −10〜−20 ／ −20〜−30 dB の数。小さい側が多いステムは、漏れを打として数えている恐れがある
        row["pseudo_levels"] = " ".join(
            f"{c[:2]}{int(np.sum(v > -10))}/{int(np.sum((v <= -10) & (v > -20)))}/{int(np.sum(v <= -20))}"
            for c, v in ((c, levels[c]) for c in ("kick", "snare", "hihat", "tom", "cymbal") if c in levels))
        from drumeval import save_hits_csv

        # 先頭のコメント行に区間とラベルのあるパートを書く（微調整 `train_transcriber.py --finetune-stems` が読む）
        save_hits_csv(ref, song_dir / "hits_pseudo.csv",
                      meta={"start_sec": f"{start:.3f}", "seconds": f"{x.size / sr:.3f}", "parts": ",".join(sorted(parts))})
    if ref is not None:
        row["ref_counts"] = _counts(ref)
        scores = transcription_scores(ref, est)
        for c in M0_CLASSES:
            row[f"f_{c}"] = scores[c].f_measure
        row["m0_1"] = float(np.mean([scores[c].f_measure for c in M0_CLASSES]))
    row["parts"] = ",".join(sorted(parts)) if parts else ""

    # 切り出し (1): 混ざったドラムから、推定した打点で
    hits = estimate_velocities(est, None, x, sr) if any(h.velocity is None for h in est) else est
    segs = cut_segments(x, sr, hits)
    kit = build_kit(segs, sr)
    row["n_hits"] = len(hits)
    row["n_isolated"] = len(segs)
    row["layers"] = ", ".join(f"{c}:{len(ls)}/{kit.n_samples(c)}" for c, ls in kit.parts.items())
    if kit.parts:
        y = _match_rms(render_kit(hits, kit, duration=x.size / sr), x)
        row["stft_rerender"] = multi_resolution_stft_distance(x, y)["total"]
        if render:
            import soundfile as sf

            sf.write(out_path("rerender_mix.wav"), (y / (np.max(np.abs(y)) + 1e-9) * 0.9).astype(np.float32), sr)

    # 切り出し (2): パート別ステムから、パートごとの打点で
    #   parts_mode="sl":   分離ツール（SpectraLayers）のパート別ステムと、その打点検出（疑似正解）
    #   parts_mode="nmfd": 推定の打点（モデル）を初期値にした NMFD で自前に分けたパート（製品の経路）
    sl_parts: dict[str, tuple[np.ndarray, int]] = {}
    if parts:
        for c, path in parts.items():
            px, psr = load_mono(str(path), sr)
            px = px[int(start * psr) :]
            sl_parts[c] = (px[: x.size], psr)
    part_audio: dict[str, tuple[np.ndarray, int]] = {}
    ref_v: list[Hit] = []
    nmfd_hits = None
    nmfd_stats: dict = {}
    if parts_mode == "nmfd" and est:
        nmfd_hits = est  # 製品の経路: モデルの打点
    elif parts_mode == "nmfd-oracle" and ref is not None:
        nmfd_hits = ref  # NMFD の上限: 疑似正解の打点（分離の質だけを見る）
    if nmfd_hits:
        from drumeval.nmfd import LEGACY, PARTS, config_for, hit_confidence, parts_for_hits, separate, snr_db

        if nmfd_legacy:  # 2026-11-15 より前の設定（比較用）: 係数が累積する縛り方、短い窓、4 パート
            ncfg = config_for(sr, residual=nmfd_residual, **LEGACY)
            nmfd_parts = PARTS
        else:
            ncfg = config_for(sr, residual=nmfd_residual, **(nmfd_options or {}))
            nmfd_parts = parts_for_hits(nmfd_hits)  # タムの打が十分あればタムのパートも作る
        sep = separate(x, sr, nmfd_hits, ncfg, parts=nmfd_parts, stats=nmfd_stats)
        part_audio = {c: (sep[c], sr) for c in nmfd_parts}
        if sl_parts:
            # パート別ステム（分離ツールか、本当のパート）を基準にした SN 比。基準が無音の種類は出さない
            row["nmfd_snr"] = " ".join(f"{c[:2]}{snr_db(sl_parts[c][0], sep[c]):+.1f}" for c in part_audio
                                       if c in sl_parts and np.sqrt(np.mean(sl_parts[c][0] ** 2)) > 1e-4)
        for c, (px, psr) in part_audio.items():
            own_hits = [h for h in nmfd_hits if to_coarse(h.cls) == c]
            if own_hits:
                ref_v += estimate_velocities(own_hits, None, px, psr)
        ref_v.sort(key=lambda h: h.time)
        if render:
            import soundfile as sf

            for c, (px, psr) in part_audio.items():
                sf.write(out_path(f"nmfd_{c}.wav"), px.astype(np.float32), psr)
            if "residual" in sep:
                sf.write(out_path("nmfd_residual.wav"), sep["residual"].astype(np.float32), sr)
    elif parts and ref is not None:
        part_audio = {c: v for c, v in sl_parts.items() if np.sqrt(np.mean(v[0] ** 2)) > 1e-5}  # 無音のパートは除く
        for c, (px, psr) in part_audio.items():
            own_hits = [h for h in ref if to_coarse(h.cls) == c]
            if own_hits:
                ref_v += estimate_velocities(own_hits, None, px, psr)
        ref_v.sort(key=lambda h: h.time)
    if part_audio and ref_v:
        # 打点を、各パートの音の立ち上がりに合わせ直す（採譜の時刻は約 10 ms 刻み。切り出す音の頭と MIDI の時刻がそろう）
        ref_v = refine_onsets(part_audio, ref_v)
        if parts_mode == "nmfd":
            # 製品の経路（`drumeval.analyzer` と同じ）: そのパートに音の無い打（採譜の誤検出）を、MIDI とキットから除く
            own0 = hit_segments_from_parts(part_audio, ref_v)
            drop = silent_hits(ref_v, [20.0 * np.log10(np.max(np.abs(a)) + 1e-9) if a is not None else None for a in own0])
            if any(drop):
                row["dropped"] = _counts([h for h, d in zip(ref_v, drop) if d])
                ref_v = [h for h, d in zip(ref_v, drop) if not d]
            if ref is not None:  # MIDI にする打点（合わせ直しと除外の後）の F 値
                sc = transcription_scores(ref, ref_v)
                row["f_midi"] = " ".join(f"{c[:2]}{sc[c].f_measure:.3f}" for c in M0_CLASSES)
        segs_p = cut_segments_from_parts(part_audio, ref_v)
        if nmfd_stats:
            # 分離の確からしさ（0〜1）。他の音と重なった打（低い値）からは、方式 C の代表音を取らない（`drumeval.analyzer` と同じ）
            conf = hit_confidence(nmfd_stats, ref_v, sr)
            attach_confidence(segs_p, ref_v, conf)
            row["confidence"] = " ".join(
                f"{c[:2]}{np.median(v):.2f}" for c in ("kick", "snare", "hihat", "tom", "cymbal")
                if (v := [cf for h, cf in zip(ref_v, conf) if to_coarse(h.cls) == c and cf is not None]))
        total = np.zeros_like(x)
        for c, (px, psr) in part_audio.items():
            total[: px.size] += px
        row["stft_parts_sum"] = multi_resolution_stft_distance(x, _match_rms(total, x))["total"]  # 分離の整合性（0 に近いほどパートの和が元に等しい）
        # 帯域ごとの差（dB）。パートの和は絶対的な差（残差に落ちた分）、再現は音量をそろえた後の差（高域が負なら「こもる」）
        row["bands_sum_db"] = band_balance_db(x, total, sr, match_rms=False)
        row["bands_sum"] = _fmt_bands(row["bands_sum_db"])
        row["parts_mode"] = parts_mode + ("" if nmfd_residual or not parts_mode.startswith("nmfd") else "（残差なし）") + (
            "（従来の設定）" if nmfd_legacy and parts_mode.startswith("nmfd") else "")
        ys = _match_rms(self_render_from_parts(part_audio, ref_v, x.size / sr), x)
        row["stft_self"] = multi_resolution_stft_distance(x, ys)["total"]  # 自己再現: 打点の誤りと区間の切り方だけの損失
        row["bands_self_db"] = band_balance_db(x, ys, sr)
        row["bands_self"] = _fmt_bands(row["bands_self_db"])
        # 各打の「自分の区間」から、元の音量と特徴（音量・減衰・明るさ）を取る
        own = hit_segments_from_parts(part_audio, ref_v)
        levels = [20.0 * np.log10(np.max(np.abs(a)) + 1e-9) if a is not None else None for a in own]
        feats = [segment_features(a, sr) if a is not None else None for a in own]
        row["n_isolated_parts"] = len(segs_p)
        # A: 3 層 × 3 音（従来）
        kit_p = build_kit(segs_p, sr)
        row["layers_parts"] = ", ".join(f"{c}:{len(ls)}/{kit_p.n_samples(c)}" for c, ls in kit_p.parts.items())
        if kit_p.parts:
            yp = _match_rms(render_kit(ref_v, kit_p, duration=x.size / sr), x)
            row["stft_rerender_parts"] = multi_resolution_stft_distance(x, yp)["total"]
            # B: 3 層 × 3 音 ＋ 打ごとの音量合わせ ＋ チョーク
            yb = _match_rms(render_kit(ref_v, kit_p, duration=x.size / sr, hit_levels_db=levels, gain_match=True, choke=True), x)
            row["stft_parts_gain_choke"] = multi_resolution_stft_distance(x, yb)["total"]
            # C: 特徴で選んだ代表 kit_size 音 ＋ 特徴の近い音を選ぶ ＋ 音量合わせ ＋ チョーク（方針 3 通り）
            kit_f = build_kit_by_features(segs_p, sr, n_per_class=parse_kit_size(kit_size))
            row["kit_features"] = ", ".join(f"{c}:{kit_f.n_samples(c)}" for c in kit_f.parts)
            renders_c = {}
            for name, policy in CHOKE_POLICIES.items():
                y_ = _match_rms(render_kit(ref_v, kit_f, duration=x.size / sr, hit_levels_db=levels, hit_features=feats, gain_match=True, choke=policy,
                                           hit_audio=own, align=True), x)
                renders_c[name] = y_
                row[f"stft_parts_features_{name}"] = multi_resolution_stft_distance(x, y_)["total"]
            row["stft_parts_features"] = row[f"stft_parts_features_{choke}"]
            yc = renders_c[choke]
            row["bands_c_db"] = band_balance_db(x, yc, sr)
            row["bands_c"] = _fmt_bands(row["bands_c_db"])
            # MIDI だけで鳴らす（プラグインの鳴らし方）: 強さ 1〜127 を各種類の打の音量の範囲に対応させ、音量の最も近い代表音を選ぶ。
            # 方式 C は打ごとに元の音の特徴で音を選ぶので、MIDI には無い情報を使っている。DAW で実際に鳴るのはこちら。
            # キットは MIDI 用（製品のキット）: 音量の順に区分し、各区分から普通の音を選ぶ（`build_kit_by_level`）
            kit_m = build_kit_by_level(segs_p, sr, n_per_class=parse_kit_size(kit_size))
            row["kit_midi"] = ", ".join(f"{c}:{kit_m.n_samples(c)}" for c in kit_m.parts)
            for c in kit_m.parts:
                kit_m.level_range[c] = level_range_for([lv for lv, h in zip(levels, ref_v) if lv is not None and to_coarse(h.cls) == c])
            hits_midi = [Hit(h.time, h.cls, velocity_for_level(lv, kit_m.level_range[to_coarse(h.cls)])
                             if lv is not None and to_coarse(h.cls) in kit_m.level_range else h.velocity) for h, lv in zip(ref_v, levels)]
            ym = _match_rms(render_midi(hits_midi, kit_m, duration=x.size / sr), x)
            row["stft_midi"] = multi_resolution_stft_distance(x, ym)["total"]
            row["bands_midi_db"] = band_balance_db(x, ym, sr)
            row["bands_midi"] = _fmt_bands(row["bands_midi_db"])
            # MIDI だけの損失の内訳: その種類だけ自己再現に置き換えたときの距離の下がり幅（どの種類が MIDI だけだと遠いか）
            y_by_cls = {c: render_midi([h for h in hits_midi if to_coarse(h.cls) == c], kit_m, duration=x.size / sr)
                        for c in sorted(part_audio) if c in kit_m.parts}
            total_m = sum(y_by_cls.values())
            shares_m = []
            for c, y_c in y_by_cls.items():
                y_self_c = self_render_from_parts({c: part_audio[c]}, [h for h in ref_v if to_coarse(h.cls) == c], x.size / sr)
                d_alt = multi_resolution_stft_distance(x, _match_rms(total_m - y_c + y_self_c, x))["total"]
                shares_m.append((c, row["stft_midi"] - d_alt))
            row["loss_shares_midi"] = " ".join(f"{c[:2]}{v:+.2f}" for c, v in sorted(shares_m, key=lambda t: -t[1]))
            # 位置の確かさ: MIDI の時刻に置いた代表音を、その打の元の音と相互相関で合わせ直すと何 ms 動くか。5 ms を超える打の割合
            # （打点の時刻と、切り出した音の頭がそろっていれば 0 に近い。方式 C は打ごとに合わせ直すので、ずれていても隠れる）。
            # 目安: 正解の時刻で確かめると、本当に 5 ms 以上ずれた打の 4〜8 割を拾い、ずれていない打の 0.2〜1.5%（ハイハットは 1〜7%）を
            # 誤って数える。低い音のキックは周期ぶん滑った位置でも相関が高いので、MIDI の位置より 1.25 倍以上良い位置だけを数える
            lead_n = int(round(SAMPLE_LEAD_SEC * sr))
            off_by = {}
            for c in kit_m.parts:
                pool = sorted(((smp, pk) for smp, pk, _ in kit_m.all_samples(c)), key=lambda t: t[1])
                peaks_c = np.array([pk for _, pk in pool])
                px = part_audio[c][0]
                lags = []
                for h, a in zip(hits_midi, own):
                    if to_coarse(h.cls) != c or a is None:
                        continue
                    smp = pool[int(np.argmin(np.abs(peaks_c - level_for_velocity(h.velocity, kit_m.level_range[c]))))][0]
                    start_h = max(int(round(h.time * sr)) - lead_n, 0)  # キットの音と同じ切り方（打点の 2 ms 前から、その打の区間の終わりまで）
                    lags.append(_onset_lag(px[start_h : start_h + lead_n + a.size], smp, sr, margin=1.25) / sr * 1000.0)
                if lags:
                    off_by[c] = float(np.mean(np.abs(np.array(lags)) > 5.0))
            row["midi_offsets"] = " ".join(f"{c[:2]}{v:.2f}" for c, v in off_by.items())
            # 代表音を 48 音に増やしたとき（既定の音の数を決める材料。MIDI は同じ）
            kit_48 = build_kit_by_level(segs_p, sr, n_per_class=48)
            kit_48.level_range = dict(kit_m.level_range)
            ym48 = _match_rms(render_midi(hits_midi, kit_48, duration=x.size / sr), x)
            row["stft_midi_48"] = multi_resolution_stft_distance(x, ym48)["total"]
            # 2026-11-17 までのキット（方式 C と同じ、特徴の k-means の代表）を、同じ MIDI で鳴らしたとき（キットの選び方の聞き比べ用）。
            # MIDI 用のキットは各音量帯の「普通の音」を選ぶので、開いたハイハットなどの明るい打の分が抜けて、超高域が少し下がる
            kit_km = replace(kit_f, level_range=dict(kit_m.level_range))
            ykm = _match_rms(render_midi(hits_midi, kit_km, duration=x.size / sr), x)
            row["stft_midi_km"] = multi_resolution_stft_distance(x, ykm)["total"]
            row["bands_midi_km_db"] = band_balance_db(x, ykm, sr)
            if parts_mode == "nmfd":
                write_hits_midi(hits_midi, out_path("hits_model.mid"))  # モデルの打点（強さは上の約束。保存したキットと合わせて DAW で鳴らせる）
            elif row.get("truth", "").startswith("疑似"):
                write_hits_midi(hits_midi, song_dir / "hits_pseudo.mid")  # 疑似正解（本当の正解がある曲では書かない）
            if render:
                import soundfile as sf

                sf.write(out_path("rerender_midi.wav"), (ym / (np.max(np.abs(ym)) + 1e-9) * 0.9).astype(np.float32), sr)
                sf.write(out_path("rerender_midi_k48.wav"), (ym48 / (np.max(np.abs(ym48)) + 1e-9) * 0.9).astype(np.float32), sr)
                sf.write(out_path("rerender_midi_km.wav"), (ykm / (np.max(np.abs(ykm)) + 1e-9) * 0.9).astype(np.float32), sr)
            # 上限: 全部の切り出し音を代表にし、自分自身だけ除いて最も近い音を選ぶ（特徴による選び方の天井）
            kit_all = build_kit_by_features(segs_p, sr, n_per_class=0, min_confidence=0.0)
            y_all = _match_rms(render_kit(ref_v, kit_all, duration=x.size / sr, hit_levels_db=levels, hit_features=feats, gain_match=True,
                                          choke=CHOKE_POLICIES[choke], hit_audio=own, align=True, leave_one_out=True), x)
            row["stft_parts_ceiling"] = multi_resolution_stft_distance(x, y_all)["total"]
            # D: 埋め込み（先頭 0.25 秒の対数メル）で代表を選び、各打に最も近い音を選ぶ ＋ 音量合わせ ＋ 位置合わせ ＋ チョーク
            embs = [head_embedding(a, sr) if a is not None else None for a in own]
            kit_e = build_kit_by_features(segs_p, sr, n_per_class=parse_kit_size(kit_size), feature_fn=head_embedding)
            yd = _match_rms(render_kit(ref_v, kit_e, duration=x.size / sr, hit_levels_db=levels, hit_features=embs, gain_match=True,
                                       choke=CHOKE_POLICIES[choke], hit_audio=own, align=True), x)
            row["stft_parts_embed"] = multi_resolution_stft_distance(x, yd)["total"]
            kit_e_all = build_kit_by_features(segs_p, sr, n_per_class=0, feature_fn=head_embedding)
            yd_all = _match_rms(render_kit(ref_v, kit_e_all, duration=x.size / sr, hit_levels_db=levels, hit_features=embs, gain_match=True,
                                           choke=CHOKE_POLICIES[choke], hit_audio=own, align=True, leave_one_out=True), x)
            row["stft_parts_embed_ceiling"] = multi_resolution_stft_distance(x, yd_all)["total"]
            if render:
                import soundfile as sf

                sf.write(out_path("rerender_parts_embed.wav"), (yd / (np.max(np.abs(yd)) + 1e-9) * 0.9).astype(np.float32), sr)
            # M0-3 を実ステムで: 疑似正解の打点を合成の汎用キットで鳴らした音を「乾いた音」とし、固定チェーンを元のステムに当てはめる
            if fit_fx:
                from drumeval.fx import apply_chain, fit_chain
                from drumeval.synth import render as render_synth

                dry = render_synth(ref_v, sr=sr, duration=x.size / sr, variation=0.0)
                params, log = fit_chain(dry, x, sr, max_seconds=fit_seconds, maxiter=3)
                wet = apply_chain(dry, sr, params)
                row["fx_before"] = multi_resolution_stft_distance(x, _match_rms(dry, x))["total"]
                row["fx_after"] = multi_resolution_stft_distance(x, _match_rms(wet, x))["total"]
                row["fx_params"] = (f"gain {params.gain_db:+.1f}dB low {params.low_shelf_db:+.1f} peak {params.peak_hz:.0f}Hz{params.peak_db:+.1f} "
                                    f"high {params.high_shelf_db:+.1f} rt60 {params.reverb_rt60:.2f}s mix {params.reverb_mix:.2f}")
                if render:
                    import soundfile as sf

                    sf.write(song_dir / "synthkit_dry.wav", (dry / (np.max(np.abs(dry)) + 1e-9) * 0.9).astype(np.float32), sr)
                    sf.write(song_dir / "synthkit_fx.wav", (wet / (np.max(np.abs(wet)) + 1e-9) * 0.9).astype(np.float32), sr)
            # 時間軸の診断: どの区間で C が悪いか、そこで自己再現も悪いか（打点側の問題か、代表音の不足か）
            tl = timeline(x, {"self": ys, "C": yc}, ref_v, sr)
            worst = sorted(tl, key=lambda r: -r["C"])[:5]
            row["worst_windows"] = "; ".join(
                f"{r['start']:.0f}-{r['end']:.0f}s C {r['C']:.2f} 自己 {r['self']:.2f} 未説明 {r['unexplained_db']:.0f}dB (ki{r['kick']} sn{r['snare']} hi{r['hihat']} cy{r['cymbal']})" for r in worst)
            # 種類ごとの損失の内訳: その種類だけ自己再現に置き換えたときの距離の下がり幅
            d_c = row["stft_parts_features"]
            shares = []
            for c in sorted(part_audio):
                keep = [i for i, h in enumerate(ref_v) if to_coarse(h.cls) != c]
                # 残りの種類は、元の方式 C と同じ鳴らし方（位置を相互相関で合わせる）。2026-11-17 まで、ここだけ位置を合わせずに
                # 鳴らしていたので、内訳が全部の種類で負になることがあった
                y_rest = render_kit([ref_v[i] for i in keep], kit_f, duration=x.size / sr, hit_levels_db=[levels[i] for i in keep],
                                    hit_features=[feats[i] for i in keep], gain_match=True, choke=CHOKE_POLICIES[choke],
                                    hit_audio=[own[i] for i in keep], align=True)
                y_self_c = self_render_from_parts({c: part_audio[c]}, [h for h in ref_v if to_coarse(h.cls) == c], x.size / sr)
                d = multi_resolution_stft_distance(x, _match_rms(y_rest + y_self_c, x))["total"]
                shares.append((c, d_c - d))
            row["loss_shares"] = " ".join(f"{c[:2]}{v:+.2f}" for c, v in sorted(shares, key=lambda t: -t[1]))
            # 種類ごとの距離: その種類だけを鳴らした音と、そのパートの音（キットを切り出した元）との距離。MIDI だけ／方式 C／自己再現。
            # 混ざった音の距離では、キックやスネアの違いはハイハットやシンバルの差に埋もれる（区分 A の a07 で、MIDI だけだと
            # キックが方式 C より悪いと聞こえたが、混ざった音の内訳ではキックの分は小さかった）
            part_d: dict[str, tuple[float, float, float]] = {}
            for c, y_c in y_by_cls.items():
                sel = [i for i, h in enumerate(ref_v) if to_coarse(h.cls) == c]
                ref_c = part_audio[c][0]
                if not sel or c not in kit_f.parts or np.sqrt(np.mean(ref_c**2)) < 1e-5:
                    continue
                y_cc = render_kit([ref_v[i] for i in sel], kit_f, duration=x.size / sr, hit_levels_db=[levels[i] for i in sel],
                                  hit_features=[feats[i] for i in sel], gain_match=True, choke=CHOKE_POLICIES[choke],
                                  hit_audio=[own[i] for i in sel], align=True)
                y_sc = self_render_from_parts({c: part_audio[c]}, [ref_v[i] for i in sel], x.size / sr)
                part_d[c] = (part_stft_distance(ref_c, y_c), part_stft_distance(ref_c, y_cc), part_stft_distance(ref_c, y_sc))
            row["part_dist_values"] = part_d
            row["part_dist"] = " ".join(f"{c[:2]}{m:.2f}/{cc:.2f}/{sf_:.2f}" for c, (m, cc, sf_) in part_d.items())
            if render:
                import csv as _csv

                with open(out_path("timeline.csv"), "w", newline="", encoding="utf-8") as f:
                    w = _csv.DictWriter(f, fieldnames=list(tl[0].keys()))
                    w.writeheader()
                    w.writerows(tl)
            if render:
                import soundfile as sf

                sf.write(out_path("rerender_parts_3x3.wav"), (yp / (np.max(np.abs(yp)) + 1e-9) * 0.9).astype(np.float32), sr)
                sf.write(out_path("rerender_parts.wav"), (yc / (np.max(np.abs(yc)) + 1e-9) * 0.9).astype(np.float32), sr)
                sf.write(out_path("rerender_self.wav"), (ys / (np.max(np.abs(ys)) + 1e-9) * 0.9).astype(np.float32), sr)
                write_audition(kit_m, out_path("kit_parts_audition.wav"))  # 製品のキット（MIDI 用）の音を、種類 → 弱→強 の順に
            if save_kit_dir:
                from drumeval.extract import save_kit

                save_kit(kit_m, song_dir / save_kit_dir)  # 製品のキット（MIDI 用）。hits_model.mid と合わせてプラグインで鳴らせる
                row["kit_saved"] = str(song_dir / save_kit_dir / "kit.json")
    if render:
        import soundfile as sf

        sf.write(song_dir / "original_cut.wav", x.astype(np.float32), sr)  # 同じ長さに切った元（聞き比べ用）
    return row


def resolve_model_path(model: str | None, song_id: str) -> str | None:
    """`--model` に `{id}` があれば曲 ID に置き換える（leave-one-out で曲ごとに別のモデルを使う: `crnn_ft_{id}.onnx`）。"""
    if not model:
        return None
    return model.replace("{id}", song_id)


def to_markdown(rows: list[dict], title: str) -> str:
    cols = ("id", "start", "seconds", "est_source", "truth", "parts_mode", "f_kick", "f_snare", "f_hihat", "m0_1", "est_counts", "ref_counts", "pseudo_levels",
            "dropped", "f_midi", "confidence",
            "n_isolated", "layers", "stft_rerender", "n_isolated_parts", "layers_parts", "stft_rerender_parts", "stft_parts_gain_choke",
            "kit_features", "stft_parts_features_all", "stft_parts_features_hihat", "stft_parts_features_none", "stft_parts_ceiling",
            "stft_parts_embed", "stft_parts_embed_ceiling", "kit_midi", "stft_midi", "stft_midi_48", "stft_midi_km", "loss_shares_midi", "midi_offsets", "part_dist", "loss_shares", "stft_self", "stft_parts_sum", "bands_sum", "bands_self", "bands_c", "bands_midi",
            "nmfd_snr", "fx_before", "fx_after", "fx_params")
    heads = ("曲ID", "開始秒", "秒", "推定", "正解の種類", "パートの出どころ", "F kick", "F snare", "F hihat", "M0-1", "推定の打（種類別）", "正解の打（種類別）",
             "疑似正解の打の大きさ（0〜−10／−10〜−20／−20〜−30 dB の数）",
             "音が無くて除いた打（--parts nmfd）", "F（MIDI にした打点。合わせ直しと除外の後）", "分離の確からしさ（種類別の中央値、0〜1）",
             "重ならない打（混合）", "層/音（混合）", "STFT（混合）", "重ならない打（パート別）", "層/音（パート別 3層×3音）", "STFT（A: 3層×3音）",
             "STFT（B: A＋音量合わせ＋チョーク）", "音の数（C: 特徴で選ぶ）", "STFT（C: チョーク全部）", "STFT（C: ハイハットだけ）", "STFT（C: チョーク無し）",
             "STFT（上限: 全部を代表・自分以外）", "STFT（D: 埋め込みで選ぶ）", "STFT（D の上限）", "音の数（MIDI 用のキット）", "STFT（MIDI だけで鳴らす）", "STFT（MIDI だけ、代表 48 音）",
             "STFT（MIDI だけ、k-means のキット）",
             "MIDI だけの損失の内訳（種類を自己再現に置き換えたときの下がり幅）", "位置が 5 ms 以上ずれる打の割合（MIDI だけ、種類別）",
             "種類ごとの距離（その種類だけを鳴らした音 vs そのパートの音。MIDI だけ／方式 C／自己再現）",
             "C の損失の内訳（種類を自己再現に置き換えたときの下がり幅）", "STFT（自己再現）", "パートの和 vs 元",
             "帯域の差 dB: パートの和 − 元（残差に落ちた分）", "帯域の差 dB: 自己再現 − 元（音量をそろえた後）", "帯域の差 dB: 方式 C − 元（音量をそろえた後）",
             "帯域の差 dB: MIDI だけで鳴らす − 元（音量をそろえた後）",
             "NMFD の SNR（パート別ステム比、dB）", "FX 当てはめ前（合成キット vs 元）", "FX 当てはめ後", "当てはめたパラメータ")
    lines = [f"# {title}", "", "| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    def f(v):
        return "" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    for r in rows:
        lines.append("| " + " | ".join(f(r.get(k)) for k in cols) + " |")
    m0_true = [r["m0_1"] for r in rows if r.get("truth") == "正解"]
    m0_pseudo = [r["m0_1"] for r in rows if r.get("truth", "").startswith("疑似")]
    lines += ["", f"- 正解付きの曲: {len(m0_true)} / {len(rows)}、疑似正解（パート別ステムの打点）付きの曲: {len(m0_pseudo)} / {len(rows)}"]
    if m0_true:
        lines.append(f"- M0-1（kick/snare/hihat の F の平均）の平均: {np.mean(m0_true):.3f}（目安 0.85）")
    if m0_pseudo:
        lines.append(f"- 疑似正解に対する M0-1 の平均: {np.mean(m0_pseudo):.3f}（分離ツールのパート分けの誤りを含むので目安。M0 の判定には使わない）")
        lines.append("- 疑似正解は、パート別ステムの包絡（その種類の帯域）が直前 60 ms の平均を 6 dB 以上超えた立ち上がり。大きい打（上位 5%）より 30 dB 以上小さいものは捨てる"
                     "（余韻の揺れ、他の種類の漏れ）。**2026-11-15 に作り直した**: それまではキックを約 3 倍に数え過ぎていたので、これより前の表の「正解の打」と F 値とは比べられない。"
                     "「疑似正解の打の大きさ」の右側（−20〜−30 dB）が多い種類は、漏れを打として数えている恐れがある。")
    d = [r["stft_rerender"] for r in rows if "stft_rerender" in r]
    if d:
        lines.append(f"- 再現との STFT 距離の平均（混ざったドラムから切り出し）: {np.mean(d):.3f}（M0-2 の聞き比べの参考。小さいほど近い）")
    dp = [r["stft_rerender_parts"] for r in rows if "stft_rerender_parts" in r]
    if dp:
        lines.append(f"- 再現との STFT 距離の平均（パート別ステムから切り出し、{len(dp)} 曲）: A（3層×3音）{np.mean(dp):.3f}、"
                     f"B（A＋音量合わせ＋チョーク）{np.mean([r['stft_parts_gain_choke'] for r in rows if 'stft_parts_gain_choke' in r]):.3f}、"
                     f"C（特徴で選ぶ代表音＋音量合わせ＋チョーク）{np.mean([r['stft_parts_features_all'] for r in rows if 'stft_parts_features_all' in r]):.3f}、"
                     f"C でチョークをハイハットだけ {np.mean([r['stft_parts_features_hihat'] for r in rows if 'stft_parts_features_hihat' in r]):.3f}、"
                     f"C でチョーク無し {np.mean([r['stft_parts_features_none'] for r in rows if 'stft_parts_features_none' in r]):.3f}、"
                     f"上限（全部を代表にして自分以外から選ぶ）{np.mean([r['stft_parts_ceiling'] for r in rows if 'stft_parts_ceiling' in r]):.3f}、"
                     f"D（埋め込みで選ぶ）{np.mean([r['stft_parts_embed'] for r in rows if 'stft_parts_embed' in r]):.3f}、"
                     f"D の上限 {np.mean([r['stft_parts_embed_ceiling'] for r in rows if 'stft_parts_embed_ceiling' in r]):.3f}")
        dm = [r["stft_midi"] for r in rows if "stft_midi" in r]
        if dm:
            d48 = [r["stft_midi_48"] for r in rows if "stft_midi_48" in r]
            lines.append(f"- 代表音を 48 音にして MIDI だけで鳴らしたときの距離の平均: {np.mean(d48):.3f}（WAV は rerender_midi_k48.wav）。")
            dkm = [r["stft_midi_km"] for r in rows if "stft_midi_km" in r]
            if dkm:
                lines.append(f"- 2026-11-17 までのキット（方式 C と同じ、特徴の k-means の代表）を MIDI だけで鳴らしたときの距離の平均: {np.mean(dkm):.3f}"
                             "（WAV は rerender_midi_km.wav）。MIDI 用のキットは各音量帯の「普通の音」を選ぶので、距離は近くなるが、開いたハイハットなどの"
                             "明るい打の分が抜けて超高域が少し下がる。聞き比べて決める。")
            lines.append(f"- **MIDI だけで鳴らしたときの距離の平均: {np.mean(dm):.3f}**（{len(dm)} 曲）。MIDI 用のキット（音量の順に区分し、各区分から普通の音を選んだもの）を、打の時刻・種類・強さだけで鳴らす（プラグインと同じ鳴らし方: "
                         "強さ 1〜127 をその種類の打の音量の範囲に対応させ、音量の最も近い代表音を選んで音量を合わせる）。方式 C は打ごとに元の音の特徴で音を選ぶので、"
                         "MIDI には無い情報を使っている。DAW で実際に鳴るのはこちら。WAV は rerender_midi.wav、MIDI は hits_model.mid。"
                         "「位置が 5 ms 以上ずれる打の割合」は、MIDI の時刻に置いた代表音を、その打の元の音に相互相関で合わせ直したときに 5 ms より大きく動く打の割合"
                         "（目安。正解の時刻で確かめると、本当にずれた打の 4〜8 割を拾い、ずれていない打も数 % 数える）。")
        pdv = [r["part_dist_values"] for r in rows if r.get("part_dist_values")]
        if pdv:
            txt = []
            for c in ("kick", "snare", "hihat", "tom", "cymbal"):
                v = np.array([p[c] for p in pdv if c in p])
                if v.size:
                    txt.append(f"{c} {v[:, 0].mean():.3f}／{v[:, 1].mean():.3f}／{v[:, 2].mean():.3f}（{v.shape[0]} 曲）")
            lines.append("- 種類ごとの距離の平均（MIDI だけ／方式 C／自己再現）: " + "、".join(txt) + "。その種類だけを鳴らした音と、そのパートの音（キットを切り出した元。"
                         "自前のパート分けならその出力）との距離。混ざった音の距離では埋もれる、キックやスネアだけの違いを見る。MIDI だけが方式 C より大きい種類は、"
                         "MIDI に無い情報（打ごとの音色と、位置の合わせ直し）に頼っていた分。")
        lines.append("- 方式 C は、特徴（音量・減衰・明るさ・帯域のエネルギー）で代表を選び、選んだ音の立ち上がりを元に相互相関で合わせて置く。"
                     "方式 D は、先頭 0.25 秒の対数メル（埋め込み）で代表を選ぶ。WAV は rerender_parts.wav（C）と rerender_parts_embed.wav（D）。")
    fx = [(r["fx_before"], r["fx_after"]) for r in rows if "fx_after" in r]
    if fx:
        lines.append(f"- M0-3（実ステム）: 合成の汎用キットで鳴らした乾いた音に、固定チェーン（ゲイン・3 バンド EQ・リバーブ）を元のステムへ当てはめた距離の平均 "
                     f"{np.mean([b for b, _ in fx]):.3f} → {np.mean([a for _, a in fx]):.3f}（{len(fx)} 曲）。WAV は synthkit_dry.wav と synthkit_fx.wav。")
    saved = [r["kit_saved"] for r in rows if r.get("kit_saved")]
    if saved:
        lines.append("- MIDI 用のキット（製品のキット）を保存した（プラグインの「kit.json を読む」で開ける）: " + "、".join(saved))
    modes = sorted({r["parts_mode"] for r in rows if r.get("parts_mode")})
    if any(m.startswith("nmfd") for m in modes):
        lines.append("- パート別の切り出しは NMFD（自前のパート分け）で行った。--parts nmfd は推定の打点、nmfd-oracle は正解（か疑似正解）の打点が初期値。"
                     "「NMFD の SNR」は、曲フォルダのパート別ステム（分離ツールの出力か、本当のパート）を基準にした SN 比（dB）。WAV は nmfd_<種類>.wav。")
    for key, label in (("bands_midi_db", "MIDI だけで鳴らす − 元"), ("bands_midi_km_db", "MIDI だけ・k-means のキット − 元"), ("bands_c_db", "方式 C − 元"), ("bands_self_db", "自己再現 − 元"), ("bands_sum_db", "パートの和 − 元")):
        bd = [r[key] for r in rows if key in r]
        if bd:
            lines.append(f"- 帯域の差の平均（{label}、{len(bd)} 曲、dB）: " + _fmt_bands({k: float(np.mean([b[k] for b in bd])) for k in bd[0]}))
    if any("bands_c_db" in r for r in rows):
        lines.append("- 帯域は 低 200 Hz 未満、中低 200 Hz〜1 kHz、中 1〜4 kHz、高 4〜10 kHz、超高 10 kHz 以上。再現（自己再現、方式 C）は音量をそろえた後の差で、"
                     "高・超高が負なら「こもる」、中低・中が負なら「痩せる」。パートの和は絶対的な差で、負の分はどのパートにも入らなかった音（NMFD の残差）。")
    lines.append("- 「開始秒」は評価に使った区間の始まり（ドラムが鳴り始める位置を自動で検出。曲フォルダの start.txt か --start で上書き）。"
                 "hits_pseudo.csv／.mid と各 WAV の時刻はこの位置からの相対。")
    lines.append("- 「種類別」の打数は ki=kick、sn=snare、hi=hihat、to=tom、cy=cymbal。")
    worst = [(r["id"], r["worst_windows"]) for r in rows if r.get("worst_windows")]
    if worst:
        lines += ["", "## 曲ごとの悪い区間（方式 C の距離が大きい 2 秒窓、上位 5。自己再現も悪ければ打点側、自己再現が良ければ代表音の不足）", ""]
        for sid, w in worst:
            lines.append(f"- {sid}: {w}")
        lines.append("- 「未説明」は、その窓で打点から再現できなかった音の割合（残差のエネルギー ÷ 元のエネルギー、dB）。0 dB に近いほど、打点に無い音（パーカッション、残響の尾など）が多い。")
        lines.append("- `--render` を付けると各曲のフォルダに `timeline.csv`（1 秒ごとの距離・未説明・打数）も書く。")
    lines.append("- STFT 距離は、比べる音の RMS を元に合わせてから測る（全体の音量差は見ない）。")
    ds = [r["stft_self"] for r in rows if "stft_self" in r]
    if ds:
        lines.append(f"- 自己再現（各打をその打自身の区間で鳴らし直す）の距離の平均: {np.mean(ds):.3f}。パート別の再現との差が、層とラウンドロビンで失う分。自己再現そのものの距離は、疑似正解の誤りと区間の切り方の分。")
    lines.append("- 「パートの和 vs 元」は、パート別ステムを足し合わせた音と元のドラムの STFT 距離。分離ツールの出力がどれだけ元を保っているかの目安で、再現の距離の下限に近い。")
    lines.append("- 疑似正解は各曲のフォルダに hits_pseudo.csv と hits_pseudo.mid（GM ドラムマップ、120 BPM の絶対時刻）で書く。DAW で元の音と並べて確かめられる。")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems_dir")
    ap.add_argument("--out", default=None, help="Markdown の出力先。省略時は標準出力")
    ap.add_argument("--est-dir", default=None, help="採譜の推定 CSV（<曲ID>.csv）の置き場")
    ap.add_argument("--max-seconds", type=float, default=60.0, help="評価に使う長さ（秒。0 で全部）")
    ap.add_argument("--start", type=float, default=None, help="評価の開始位置（秒）。省略時はドラムが鳴り始める位置を自動で検出。曲ごとに変えるなら曲フォルダに start.txt")
    ap.add_argument("--model", default=None,
                    help="ONNX の採譜モデル（指定すると基準値の代わりに使う。onnxruntime が要る）。`{id}` を含めると曲 ID に置き換える"
                         "（leave-one-out の微調整モデルを曲ごとに使う。例: \"analysis\\models\\crnn_ft_v5_{id}.onnx\"。無い曲は飛ばす）")
    ap.add_argument("--render", action="store_true", help="聞き比べ用の WAV（original_cut / rerender_midi / rerender_midi_km / rerender_midi_k48 / rerender_parts / rerender_parts_3x3 / rerender_mix / rerender_self / kit_parts_audition）を各曲のフォルダに書く")
    ap.add_argument("--threshold", type=float, default=None, help="--model の打点のしきい値（既定はモデルの設定、0.5）")
    ap.add_argument("--kit-size", default="24", help="キット（MIDI 用と方式 C）で種類ごとに残す音の数（既定 24）。種類別なら kick=48,snare=24,hihat=24,cymbal=24")
    ap.add_argument("--choke", choices=sorted(CHOKE_POLICIES), default="all", help="rerender_parts.wav に使うチョークの方針（表には 3 通りとも出る）")
    ap.add_argument("--only", default=None, help="曲 ID をコンマ区切りで指定すると、その曲だけ評価する（例: a01,a07）")
    ap.add_argument("--parts", choices=("sl", "nmfd", "nmfd-oracle"), default="sl",
                    help="パート別の切り出しに使うパート: sl = 分離ツールのパート別ステムと疑似正解、nmfd = 推定の打点を初期値にした NMFD（製品の経路）、"
                         "nmfd-oracle = 疑似正解の打点を初期値にした NMFD（分離の質だけを見る上限）")
    ap.add_argument("--nmfd-no-residual", action="store_true", help="NMFD で残差の成分を使わない（パートの和を元に等しくする）")
    ap.add_argument("--tag", default="",
                    help="--render などで書き出すファイルの名前の末尾に付ける印（英数字と _ -）。例: --tag v5 で rerender_parts_v5.wav。"
                         "モデルを替えた結果を上書きせずに並べて聞ける（original_cut.wav には付かない）")
    ap.add_argument("--nmfd-legacy", action="store_true",
                    help="NMFD を 2026-11-15 より前の設定で動かす（比較用。打点の後の音がパートから消え、再現がこもる）")
    ap.add_argument("--save-kit", default=None, metavar="DIR",
                    help="MIDI 用のキット（製品のキット。音の WAV と kit.json）を各曲のフォルダの DIR に保存する（プラグインで読める。例: kit）")
    ap.add_argument("--fit-fx", action="store_true", help="M0-3 を実ステムで測る: 合成キットの乾いた音に固定チェーンを当てはめ、前後の距離を出す（1 曲 2〜3 分）")
    ap.add_argument("--fit-seconds", type=float, default=20.0, help="当てはめに使う長さ（秒）")
    args = ap.parse_args(argv)
    if args.tag and not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        ap.error("--tag は英数字と _ - だけにする")

    stems_dir = Path(args.stems_dir)
    if not stems_dir.exists():
        print(f"フォルダが無い: {stems_dir.resolve()}", file=sys.stderr)
        return 1
    songs = list_songs(stems_dir)
    if args.only:
        wanted = {w.strip() for w in args.only.split(",") if w.strip()}
        songs = [s_ for s_ in songs if s_[0] in wanted]
    if not songs:
        print(f"音声ファイルが見つからない: {stems_dir.resolve()}", file=sys.stderr)
        print("  期待する形: data/stems/a01/drums.wav（フォルダ内の音声が 1 つなら名前は問わない）か、data/stems/a01.wav", file=sys.stderr)
        for p in sorted(stems_dir.iterdir()):
            print("  見つかったもの:", p.name, "(フォルダ)" if p.is_dir() else "", file=sys.stderr)
        return 1
    rows = []
    for song_id, d, f in songs:
        est = Path(args.est_dir) / f"{song_id}.csv" if args.est_dir else None
        model = resolve_model_path(args.model, song_id)
        if model and not Path(model).exists():
            print(f"{song_id}: モデルが無い（{model}）。飛ばす", file=sys.stderr)
            continue
        print(f"{song_id} ({f.name}) ...", file=sys.stderr)
        rows.append(evaluate_song(d, est, args.max_seconds, song_id=song_id, wav=f, model=model, render=args.render, threshold=args.threshold, kit_size=args.kit_size, choke=args.choke, start_sec=args.start,
                                  fit_fx=args.fit_fx, fit_seconds=args.fit_seconds, parts_mode=args.parts,
                                  save_kit_dir=args.save_kit, nmfd_residual=not args.nmfd_no_residual, nmfd_legacy=args.nmfd_legacy, tag=args.tag))
    md = to_markdown(rows, f"評価の結果（{date.today().isoformat()}）")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md, encoding="utf-8")
        print("書き出した:", args.out, file=sys.stderr)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
