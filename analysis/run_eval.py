#!/usr/bin/env python3
"""評価用ステムの集まり（`data/stems/<曲ID>/`）に対して、M0 の3指標をまとめて出す（回帰テスト）。

使い方::

    python run_eval.py data/stems [--out docs/phase0/結果_YYYY-MM-DD.md] [--est-dir 推定の置き場] [--max-seconds 60]
                       [--model analysis/models/crnn_synth_v0.onnx] [--render]

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
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drumeval import Hit, load_hits, multi_resolution_stft_distance, transcription_scores  # noqa: E402
from drumeval.baseline import onsets_from_part, transcribe  # noqa: E402
from drumeval.extract import build_kit, cut_segments, cut_segments_from_parts, estimate_velocities, render_kit, self_render_from_parts  # noqa: E402
from drumeval.labels import M0_CLASSES, to_coarse  # noqa: E402
from stft_distance import load_mono  # noqa: E402


AUDIO_EXT = (".wav", ".flac", ".aif", ".aiff", ".ogg")


def find_stem(song_dir: Path) -> Path | None:
    """`drums.wav`／`drums.flac` を優先し、無ければフォルダ内の音声ファイルが 1 つだけならそれを使う。"""
    for name in ("drums.wav", "drums.flac", "drum.wav", "drum.flac", "ドラム.wav"):
        if (song_dir / name).exists():
            return song_dir / name
    # SpectraLayers の書き出し「<曲名>_ドラム.wav」
    sl = [p for p in song_dir.iterdir() if p.suffix.lower() in AUDIO_EXT and p.stem.endswith("_ドラム")]
    if len(sl) == 1:
        return sl[0]
    audio = [p for p in song_dir.iterdir() if p.suffix.lower() in AUDIO_EXT and not p.name.startswith("rerender")]
    return audio[0] if len(audio) == 1 else None


# パート別に分離されたステムの名前（SpectraLayers の日本語名と英語名）→ 種類
PART_NAMES = {
    "キック": "kick", "kick": "kick", "kicks": "kick",
    "スネア": "snare", "snare": "snare",
    "ハイハット": "hihat", "hihat": "hihat", "hi-hat": "hihat", "hats": "hihat",
    "シンバル": "cymbal", "cymbal": "cymbal", "cymbals": "cymbal",
    "タム": "tom", "tom": "tom", "toms": "tom",
}


def find_part_stems(song_dir: Path) -> dict[str, Path]:
    """パート別ステム（キック.wav など）があれば 種類 → ファイル を返す。"""
    out: dict[str, Path] = {}
    for p in song_dir.iterdir():
        if p.suffix.lower() not in AUDIO_EXT:
            continue
        key = p.stem.lower()
        for name, cls in PART_NAMES.items():
            if key == name.lower() or key.endswith("_" + name.lower()):
                out[cls] = p
    return out


def pseudo_truth_from_parts(parts: dict[str, Path], max_seconds: float, sr_target: int | None) -> list[Hit]:
    """パート別ステムの打点検出を「疑似的な正解」にする。分離の誤りはそのまま残るので目安。"""
    hits: list[Hit] = []
    for cls, path in parts.items():
        x, sr = load_mono(str(path), sr_target)
        if max_seconds and x.size > max_seconds * sr:
            x = x[: int(max_seconds * sr)]
        for t in onsets_from_part(x, sr):
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


def write_hits_midi(hits: list[Hit], path: Path, bpm: float = 120.0) -> None:
    """打点を GM ドラムマップの MIDI にする（DAW に読み込んで元の音と並べて確かめる用）。"""
    import mido

    note_of = {"kick": 36, "snare": 38, "hihat": 42, "tom": 45, "cymbal": 49}
    mid = mido.MidiFile(type=0, ticks_per_beat=480)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    events = []
    for h in hits:
        tick = int(round(h.time * bpm / 60.0 * 480))
        note = note_of.get(to_coarse(h.cls), 39)
        vel = h.velocity or 100
        events.append((tick, 1, mido.Message("note_on", channel=9, note=note, velocity=vel, time=0)))
        events.append((tick + 48, 0, mido.Message("note_off", channel=9, note=note, velocity=0, time=0)))
    events.sort(key=lambda e: (e[0], e[1]))
    last = 0
    for tick, _, msg in events:
        msg.time = tick - last
        last = tick
        tr.append(msg)
    mid.save(str(path))


def _match_rms(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """y の RMS を x に合わせる（全体の音量差で STFT 距離が膨らまないようにする）。"""
    rx = np.sqrt(np.mean(x**2)) + 1e-9
    ry = np.sqrt(np.mean(y**2)) + 1e-9
    return y * (rx / ry)


def _counts(hits: list[Hit]) -> str:
    from collections import Counter

    c = Counter(to_coarse(h.cls) for h in hits)
    return " ".join(f"{k[:2]}{c[k]}" for k in ("kick", "snare", "hihat", "tom", "cymbal") if c.get(k))


def evaluate_song(song_dir: Path, est_csv: Path | None, max_seconds: float, song_id: str | None = None, wav: Path | None = None,
                  model: str | None = None, render: bool = False, threshold: float | None = None) -> dict:
    wav = wav or find_stem(song_dir)
    x, sr = load_mono(str(wav), None)
    if max_seconds and x.size > max_seconds * sr:
        x = x[: int(max_seconds * sr)]
    row: dict = {"id": song_id or song_dir.name, "seconds": x.size / sr}

    if est_csv and est_csv.exists():
        est = load_hits(est_csv)
        row["est_source"] = est_csv.name
    elif model:
        from drumeval.infer import transcribe_onnx

        est = transcribe_onnx(x, sr, model, threshold)
        row["est_source"] = Path(model).stem + ("" if threshold is None else f"@{threshold:g}")
    else:
        est = transcribe(x, sr)
        row["est_source"] = "baseline"
    est = [h for h in est if h.time < x.size / sr]
    row["est_counts"] = _counts(est)

    truth_path = next((p for p in (song_dir / "hits.csv", song_dir / "hits.mid", song_dir / "hits.midi") if p.exists()), None)
    parts = find_part_stems(song_dir)
    ref: list[Hit] | None = None
    if truth_path:
        ref = [h for h in load_hits(truth_path) if h.time < x.size / sr]
        row["truth"] = "正解"
    elif all(c in parts for c in M0_CLASSES):
        ref = pseudo_truth_from_parts(parts, x.size / sr, sr)  # シンバル・タムのステムがあればそれも含める
        row["truth"] = "疑似（パート別ステム）"
        from drumeval import save_hits_csv

        save_hits_csv(ref, song_dir / "hits_pseudo.csv")
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

            sf.write(song_dir / "rerender_mix.wav", (y / (np.max(np.abs(y)) + 1e-9) * 0.9).astype(np.float32), sr)

    # 切り出し (2): パート別ステムから、パートごとの打点（疑似正解）で
    if parts and ref is not None and row.get("truth", "").startswith("疑似"):
        part_audio: dict[str, tuple[np.ndarray, int]] = {}
        for c, path in parts.items():
            px, psr = load_mono(str(path), sr)
            part_audio[c] = (px[: x.size], psr)
        ref_v: list[Hit] = []
        for c, (px, psr) in part_audio.items():
            own = [h for h in ref if to_coarse(h.cls) == c]
            if own:
                ref_v += estimate_velocities(own, None, px, psr)
        ref_v.sort(key=lambda h: h.time)
        segs_p = cut_segments_from_parts(part_audio, ref_v)
        kit_p = build_kit(segs_p, sr)
        total = np.zeros_like(x)
        for c, (px, psr) in part_audio.items():
            total[: px.size] += px
        row["stft_parts_sum"] = multi_resolution_stft_distance(x, _match_rms(total, x))["total"]  # 分離の整合性（0 に近いほどパートの和が元に等しい）
        write_hits_midi(ref_v, song_dir / "hits_pseudo.mid")  # DAW で確かめる用
        ys = _match_rms(self_render_from_parts(part_audio, ref_v, x.size / sr), x)
        row["stft_self"] = multi_resolution_stft_distance(x, ys)["total"]  # 自己再現: 打点の誤りと区間の切り方だけの損失
        if render:
            import soundfile as sf

            sf.write(song_dir / "rerender_self.wav", (ys / (np.max(np.abs(ys)) + 1e-9) * 0.9).astype(np.float32), sr)
        row["n_isolated_parts"] = len(segs_p)
        row["layers_parts"] = ", ".join(f"{c}:{len(ls)}/{kit_p.n_samples(c)}" for c, ls in kit_p.parts.items())
        if kit_p.parts:
            yp = _match_rms(render_kit(ref_v, kit_p, duration=x.size / sr), x)
            row["stft_rerender_parts"] = multi_resolution_stft_distance(x, yp)["total"]
            if render:
                import soundfile as sf

                sf.write(song_dir / "rerender_parts.wav", (yp / (np.max(np.abs(yp)) + 1e-9) * 0.9).astype(np.float32), sr)
    if render:
        import soundfile as sf

        sf.write(song_dir / "original_cut.wav", x.astype(np.float32), sr)  # 同じ長さに切った元（聞き比べ用）
    return row


def to_markdown(rows: list[dict], title: str) -> str:
    cols = ("id", "seconds", "est_source", "truth", "f_kick", "f_snare", "f_hihat", "m0_1", "est_counts", "ref_counts",
            "n_isolated", "layers", "stft_rerender", "n_isolated_parts", "layers_parts", "stft_rerender_parts", "stft_self", "stft_parts_sum")
    heads = ("曲ID", "秒", "推定", "正解の種類", "F kick", "F snare", "F hihat", "M0-1", "推定の打（種類別）", "正解の打（種類別）",
             "重ならない打（混合）", "層/音（混合）", "STFT 距離（混合）", "重ならない打（パート別）", "層/音（パート別）", "STFT 距離（パート別）", "STFT 距離（自己再現）", "パートの和 vs 元")
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
    d = [r["stft_rerender"] for r in rows if "stft_rerender" in r]
    if d:
        lines.append(f"- 再現との STFT 距離の平均（混ざったドラムから切り出し）: {np.mean(d):.3f}（M0-2 の聞き比べの参考。小さいほど近い）")
    dp = [r["stft_rerender_parts"] for r in rows if "stft_rerender_parts" in r]
    if dp:
        lines.append(f"- 再現との STFT 距離の平均（パート別ステムから切り出し、{len(dp)} 曲）: {np.mean(dp):.3f}")
    lines.append("- 「種類別」の打数は ki=kick、sn=snare、hi=hihat、to=tom、cy=cymbal。")
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
    ap.add_argument("--max-seconds", type=float, default=60.0, help="曲の先頭から使う長さ（0 で全部）")
    ap.add_argument("--model", default=None, help="ONNX の採譜モデル（指定すると基準値の代わりに使う。onnxruntime が要る）")
    ap.add_argument("--render", action="store_true", help="聞き比べ用の WAV（original_cut / rerender_mix / rerender_parts / rerender_self）を各曲のフォルダに書く")
    ap.add_argument("--threshold", type=float, default=None, help="--model の打点のしきい値（既定はモデルの設定、0.5）")
    args = ap.parse_args(argv)

    stems_dir = Path(args.stems_dir)
    if not stems_dir.exists():
        print(f"フォルダが無い: {stems_dir.resolve()}", file=sys.stderr)
        return 1
    songs = list_songs(stems_dir)
    if not songs:
        print(f"音声ファイルが見つからない: {stems_dir.resolve()}", file=sys.stderr)
        print("  期待する形: data/stems/a01/drums.wav（フォルダ内の音声が 1 つなら名前は問わない）か、data/stems/a01.wav", file=sys.stderr)
        for p in sorted(stems_dir.iterdir()):
            print("  見つかったもの:", p.name, "(フォルダ)" if p.is_dir() else "", file=sys.stderr)
        return 1
    rows = []
    for song_id, d, f in songs:
        est = Path(args.est_dir) / f"{song_id}.csv" if args.est_dir else None
        print(f"{song_id} ({f.name}) ...", file=sys.stderr)
        rows.append(evaluate_song(d, est, args.max_seconds, song_id=song_id, wav=f, model=args.model, render=args.render, threshold=args.threshold))
    md = to_markdown(rows, f"評価の結果（{date.today().isoformat()}）")
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print("書き出した:", args.out, file=sys.stderr)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
