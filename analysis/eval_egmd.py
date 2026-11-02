#!/usr/bin/env python3
"""ONNX の採譜モデルを E-GMD の分割（既定 test）で測る。正解は MIDI なので、これが本当の F 値（電子ドラムの音に対して）。

使い方::

    python eval_egmd.py models/crnn_egmd_v1.onnx D:\\data\\e-gmd-v1.0.0 [--split test] [--max-tracks 60] [--seconds 30] [--check-offset]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drumeval import transcription_scores  # noqa: E402
from drumeval.egmd import estimate_offset, load_audio, read_index  # noqa: E402
from drumeval.features import log_mel, pick_hits  # noqa: E402
from drumeval.infer import load_meta  # noqa: E402
from drumeval.labels import M0_CLASSES, load_hits_midi  # noqa: E402


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
        others = [h for k, hs in ref_by.items() if k in ("kick", "snare") for h in hs]
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
    ap.add_argument("--check-offset", action="store_true", help="音声と MIDI のずれの見積もりも出す")
    ap.add_argument("--by-kit", action="store_true", help="キットごとの M0-1 も出す（電子系のキットが弱いかを見る）")
    ap.add_argument("--diagnose", action="store_true", help="ハイハットとシンバルの取りこぼし・誤検出の内訳（小さい打か、同時の打か、混同か）を出す")
    ap.add_argument("--sweep", action="store_true", help="しきい値 0.3〜0.8 で種類ごとの F を出す")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import onnxruntime as ort

    cfg, classes, thr = load_meta(args.model)
    thr = args.threshold if args.threshold is not None else thr
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    tracks = [t for t in read_index(args.egmd) if t.split == args.split]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(tracks)  # type: ignore[arg-type]
    tracks = tracks[: args.max_tracks]
    if not tracks:
        print("トラックが無い", file=sys.stderr)
        return 1

    per_class = {c: {"p": [], "r": [], "f": []} for c in classes}
    offsets = []
    by_kit: dict[str, list[float]] = {}
    sweep: dict[float, dict[str, list[float]]] = {t: {c: [] for c in classes} for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)}
    diag = {c: {"fn_quiet": 0, "fn_loud": 0, "fn_masked": 0, "fn_free": 0, "fp_other": {}, "fp_none": 0, "n_ref": 0, "n_est": 0} for c in ("hihat", "cymbal", "snare")}
    for tr in tracks:
        x = load_audio(tr.audio, cfg.sr, 0.0, args.seconds)
        hits = [h for h in load_hits_midi(tr.midi) if h.time < x.size / cfg.sr]
        prob = sess.run(None, {sess.get_inputs()[0].name: log_mel(x, cfg)[None]})[0][0]
        est = pick_hits(prob, cfg, classes, thr)
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
        by_kit.setdefault(tr.kit or "?", []).append(float(np.mean([s[c].f_measure for c in M0_CLASSES if c in s])))
        if args.check_offset:
            offsets.append(estimate_offset(x, cfg.sr, hits, cfg))

    print(f"E-GMD {args.split} 分割 {len(tracks)} 本 × {args.seconds:.0f} 秒、しきい値 {thr}（特徴量: {cfg.n_mels} メル）")
    print(f"{'種類':8s} {'P':>6s} {'R':>6s} {'F':>6s}")
    for c in classes:
        print(f"{c:8s} {np.mean(per_class[c]['p']):6.3f} {np.mean(per_class[c]['r']):6.3f} {np.mean(per_class[c]['f']):6.3f}")
    m0 = float(np.mean([np.mean(per_class[c]["f"]) for c in M0_CLASSES if c in per_class]))
    print(f"M0-1（kick/snare/hihat の F の平均）: {m0:.3f}  目安 0.85")
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
