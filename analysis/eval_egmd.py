#!/usr/bin/env python3
"""ONNX の採譜モデルを E-GMD の分割（既定 test）で測る。正解は MIDI なので、これが本当の F 値（電子ドラムの音に対して）。

使い方::

    python eval_egmd.py models/crnn_egmd_v1.onnx D:\\data\\e-gmd-v1.0.0 [--split test] [--max-tracks 60] [--seconds 30] [--no-label-offsets]
                        [--by-kit] [--worst 10] [--diagnose] [--sweep] [--min-velocity 40]

E-GMD の正解の MIDI は、ファイルごとに音とずれていることがある（test 分割の 9% が 50 ms 以上 = 採点の許容を超える）。
`egmd_offsets.py` で測った一覧（`label_offsets.csv`）が E-GMD のフォルダにあれば、ずれを直した正解で採点する。
`--no-label-offsets` で直さない（v5 までの数字はこの状態）。両方を出して比べると、ずれの影響が分かる
（v5: 直さない正解で M0-1 0.773、直した正解で 0.819）。

出力:
- 種類ごとの P／R／F（曲ごとの値の平均。正解にも推定にも打の無い曲は 1 と数えるので、タムやシンバルは高く出る）と M0-1。
- 打をまとめた値（全部の曲の打を足して数える）と、正解の打の数、打のある曲の数。
- 正解のずれが 10 ms 以上の曲と 10 ms 未満の曲の M0-1（直した後も「10 ms 以上」の側だけ低ければ、直し方が合っていない疑い）。
- `--worst N`: M0-1 の低い曲を N 本、ずれ・確かさ・キットと一緒に。`--by-kit`: キットごとの M0-1。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drumeval import transcription_scores  # noqa: E402
from drumeval.align import MIN_SHARPNESS  # noqa: E402
from drumeval.egmd import estimate_offset, find_index_csv, load_audio, read_index, read_offsets, track_hits  # noqa: E402
from drumeval.features import log_mel, pick_hits  # noqa: E402
from drumeval.infer import import_onnxruntime, load_meta  # noqa: E402
from drumeval.labels import M0_CLASSES  # noqa: E402


def _drop_quiet(ref, est, min_velocity: int, tol: float = 0.05):
    """velocity が min_velocity 未満の正解の打と、それに対応する（同じ種類で ±tol 秒の）推定の打を外す。"""
    if min_velocity <= 0:
        return ref, est
    from drumeval.labels import to_coarse

    quiet = [h for h in ref if (h.velocity or 127) < min_velocity]
    keep_ref = [h for h in ref if (h.velocity or 127) >= min_velocity]
    keep_est = [g for g in est if not any(to_coarse(q.cls) == to_coarse(g.cls) and abs(q.time - g.time) <= tol for q in quiet)]
    return keep_ref, keep_est


def _diagnose(ref, est, diag: dict, tol: float = 0.05) -> None:
    from drumeval.labels import to_coarse

    ref_by = {}
    for h in ref:
        ref_by.setdefault(to_coarse(h.cls), []).append(h)
    est_by = {}
    for h in est:
        est_by.setdefault(to_coarse(h.cls), []).append(h)
    for c, d in diag.items():
        r = sorted(ref_by.get(c, []), key=lambda h: h.time)
        e = sorted(est_by.get(c, []), key=lambda h: h.time)
        d["n_ref"] += len(r)
        d["n_est"] += len(e)
        used = set()
        matched_ref = set()
        for i, h in enumerate(r):  # 貪欲な対応付け
            best, best_j = tol + 1, None
            for j, g in enumerate(e):
                if j in used:
                    continue
                dt = abs(g.time - h.time)
                if dt <= tol and dt < best:
                    best, best_j = dt, j
            if best_j is not None:
                used.add(best_j)
                matched_ref.add(i)
        others = [h for k, hs in ref_by.items() if k in ("kick", "snare") and k != c for h in hs]
        for i, h in enumerate(r):
            if i in matched_ref:
                continue
            if (h.velocity or 127) < 40:
                d["fn_quiet"] += 1
            else:
                d["fn_loud"] += 1
            if any(abs(o.time - h.time) <= 0.03 for o in others):
                d["fn_masked"] += 1
            else:
                d["fn_free"] += 1
        for j, g in enumerate(e):
            if j in used:
                continue
            near = [k for k, hs in ref_by.items() if k != c and any(abs(o.time - g.time) <= tol for o in hs)]
            if near:
                d["fp_other"][near[0]] = d["fp_other"].get(near[0], 0) + 1
            else:
                d["fp_none"] += 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("egmd")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-tracks", type=int, default=60)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--check-offset", action="store_true", help="音声と MIDI のずれの見積もりも出す（粗い測り方。正確な一覧は egmd_offsets.py）")
    ap.add_argument("--no-label-offsets", action="store_true", help="正解の時刻のずれ（label_offsets.csv）を直さずに採点する（比較用）")
    ap.add_argument("--by-kit", action="store_true", help="キットごとの M0-1 も出す（電子系のキットが弱いかを見る）")
    ap.add_argument("--worst", type=int, default=0, metavar="N", help="M0-1 の低い曲を N 本、ずれ・確かさ・キットと一緒に出す（ずれの直し方が合っていない曲を探す）")
    ap.add_argument("--diagnose", action="store_true", help="ハイハットとシンバルの取りこぼし・誤検出の内訳（小さい打か、同時の打か、混同か）を出す")
    ap.add_argument("--sweep", action="store_true", help="しきい値 0.3〜0.8 で種類ごとの F を出す")
    ap.add_argument("--min-velocity", type=int, default=0, help="この velocity 未満の正解の打を評価から外す（例: 40。ゴーストノートを除いた F 値）。"
                    "外した打に対応する推定（±50 ms）も数えない")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    ort = import_onnxruntime()

    cfg, classes, thr = load_meta(args.model)
    thr = args.threshold if args.threshold is not None else thr
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    tracks = [t for t in read_index(args.egmd, use_offsets=not args.no_label_offsets) if t.split == args.split]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(tracks)  # type: ignore[arg-type]
    tracks = tracks[: args.max_tracks]
    if not tracks:
        print("トラックが無い", file=sys.stderr)
        return 1

    # ずれの一覧は、採点に使わないとき（--no-label-offsets）も読む（ずれの大きかった曲と小さかった曲に分けて見るため）
    measured = read_offsets(find_index_csv(args.egmd).parent)
    per_class = {c: {"p": [], "r": [], "f": []} for c in classes}
    pooled = {c: {"ref": 0, "est": 0, "hit": 0, "tracks": 0} for c in classes}  # 全部の曲の打を足した数（hit は対応の付いた打）
    by_offset: dict[str, list[float]] = {"shifted": [], "aligned": []}
    per_track: list[tuple[float, str]] = []
    offsets = []
    by_kit: dict[str, list[float]] = {}
    sweep: dict[float, dict[str, list[float]]] = {t: {c: [] for c in classes} for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)}
    diag = {c: {"fn_quiet": 0, "fn_loud": 0, "fn_masked": 0, "fn_free": 0, "fp_other": {}, "fp_none": 0, "n_ref": 0, "n_est": 0} for c in ("hihat", "cymbal", "snare")}
    for tr in tracks:
        x = load_audio(tr.audio, cfg.sr, 0.0, args.seconds)
        hits_all = [h for h in track_hits(tr) if h.time < x.size / cfg.sr]  # ずれの一覧があれば、直した時刻
        prob = sess.run(None, {sess.get_inputs()[0].name: log_mel(x, cfg)[None]})[0][0]
        est_all = pick_hits(prob, cfg, classes, thr)
        hits, est = _drop_quiet(hits_all, est_all, args.min_velocity)
        if args.sweep:
            for t in sweep:
                s_t = transcription_scores(hits, pick_hits(prob, cfg, classes, t), classes)
                for c in classes:
                    sweep[t][c].append(s_t[c].f_measure)
        if args.diagnose:
            _diagnose(hits, est, diag)
        s = transcription_scores(hits, est, classes)
        for c in classes:
            per_class[c]["p"].append(s[c].precision)
            per_class[c]["r"].append(s[c].recall)
            per_class[c]["f"].append(s[c].f_measure)
            pooled[c]["ref"] += s[c].n_ref
            pooled[c]["est"] += s[c].n_est
            pooled[c]["tracks"] += int(s[c].n_ref > 0)
            if s[c].n_ref and s[c].n_est:
                pooled[c]["hit"] += int(round(s[c].recall * s[c].n_ref))
        m0_track = float(np.mean([s[c].f_measure for c in M0_CLASSES if c in s]))
        by_kit.setdefault(tr.kit or "?", []).append(m0_track)
        off_ms, sharp = (measured[tr.key][0] * 1000.0, measured[tr.key][1]) if tr.key in measured else (None, 0.0)
        if off_ms is not None and sharp >= MIN_SHARPNESS:
            by_offset["shifted" if abs(off_ms) >= 10.0 else "aligned"].append(m0_track)
        per_track.append((m0_track, f"ずれ {'未測定' if off_ms is None else f'{off_ms:+.0f} ms（確かさ {sharp:.1f}）'}  {tr.kit or '?'}  {tr.key}"))
        if args.check_offset:
            offsets.append(estimate_offset(x, cfg.sr, hits, cfg))

    n_shifted = sum(1 for t in tracks if abs(t.offset) >= 0.005)
    print("正解の時刻のずれ: " + ("直さない（--no-label-offsets）" if args.no_label_offsets else
                           f"{n_shifted} / {len(tracks)} 本を 5 ms 以上直した（label_offsets.csv）" if n_shifted else
                           "直していない（label_offsets.csv が無い。python analysis/egmd_offsets.py <E-GMD のフォルダ> で作る）"))
    print(f"E-GMD {args.split} 分割 {len(tracks)} 本 × {args.seconds:.0f} 秒、しきい値 {thr}（特徴量: {cfg.n_mels} メル）"
          + (f"、velocity {args.min_velocity} 未満の打を除く" if args.min_velocity > 0 else ""))
    print(f"{'種類':8s} {'P':>6s} {'R':>6s} {'F':>6s}")
    for c in classes:
        print(f"{c:8s} {np.mean(per_class[c]['p']):6.3f} {np.mean(per_class[c]['r']):6.3f} {np.mean(per_class[c]['f']):6.3f}")
    m0 = float(np.mean([np.mean(per_class[c]["f"]) for c in M0_CLASSES if c in per_class]))
    print(f"M0-1（kick/snare/hihat の F の平均）: {m0:.3f}  目安 0.85")
    # 上の表は曲ごとの値の平均で、正解にも推定にも打の無い曲は 1 と数える（タムやシンバルは、打の無い曲が大半を占める）。
    # 下は全部の曲の打を足して数えた値
    print("打をまとめた値（全部の曲の打を足して数える。打の無い曲を 1 と数えない）:")
    print(f"{'種類':8s} {'P':>6s} {'R':>6s} {'F':>6s}  正解の打  打のある曲")
    pooled_f = {}
    for c in classes:
        d = pooled[c]
        pp = d["hit"] / d["est"] if d["est"] else 0.0
        rr = d["hit"] / d["ref"] if d["ref"] else 0.0
        pooled_f[c] = 2 * pp * rr / (pp + rr) if pp + rr > 0 else 0.0
        print(f"{c:8s} {pp:6.3f} {rr:6.3f} {pooled_f[c]:6.3f}  {d['ref']:8d}  {d['tracks']:4d} / {len(tracks)}")
    print(f"まとめた M0-1: {float(np.mean([pooled_f[c] for c in M0_CLASSES if c in pooled_f])):.3f}")
    if by_offset["shifted"] or by_offset["aligned"]:
        def _mean(v: list[float]) -> str:
            return f"{np.mean(v):.3f}（{len(v)} 本）" if v else "（0 本）"

        print(f"正解のずれ（label_offsets.csv）が 10 ms 以上の曲の M0-1: {_mean(by_offset['shifted'])}、10 ms 未満の曲: {_mean(by_offset['aligned'])}"
              "  ← 直した後も「10 ms 以上」の側だけ低ければ、ずれの直し方が合っていない疑い（--worst 10 で曲を見る）")
    if args.worst > 0:
        print(f"M0-1 の低い曲（下位 {args.worst}）:")
        for v, text in sorted(per_track, key=lambda t: t[0])[: args.worst]:
            print(f"  {v:.3f}  {text}")
    if offsets:
        print(f"音声と MIDI のずれの見積もり: 中央値 {np.median(offsets) * 1000:.1f} ms（正なら音声が遅い）")
    if args.by_kit:
        print("キットごとの M0-1（本数）:")
        for kit, vals in sorted(by_kit.items(), key=lambda kv: np.mean(kv[1])):
            print(f"  {np.mean(vals):.3f}  ({len(vals)})  {kit}")
    if args.sweep:
        print("しきい値ごとの F:")
        for t, d in sweep.items():
            print(f"  thr {t:.1f}: " + " ".join(f"{c} {np.mean(d[c]):.3f}" for c in classes))
    if args.diagnose:
        print("誤りの内訳（正解は MIDI。小さい打 = velocity < 40、同時 = 30 ms 以内にキックかスネア）:")
        for c, d in diag.items():
            fn = d["fn_quiet"] + d["fn_loud"]
            fp = sum(d["fp_other"].values()) + d["fp_none"]
            print(f"  {c}: 正解 {d['n_ref']} 打、推定 {d['n_est']} 打、取りこぼし {fn}（小さい打 {d['fn_quiet']}、大きい打 {d['fn_loud']}；"
                  f"同時の打 {d['fn_masked']}、単独 {d['fn_free']}）、誤検出 {fp}（"
                  + ", ".join(f"{k} の打点と重なる {v}" for k, v in sorted(d["fp_other"].items(), key=lambda kv: -kv[1])) + f", 何も無い {d['fp_none']}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
