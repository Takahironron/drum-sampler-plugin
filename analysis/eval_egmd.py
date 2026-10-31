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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("egmd")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-tracks", type=int, default=60)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--check-offset", action="store_true", help="音声と MIDI のずれの見積もりも出す")
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
    for tr in tracks:
        x = load_audio(tr.audio, cfg.sr, 0.0, args.seconds)
        hits = [h for h in load_hits_midi(tr.midi) if h.time < x.size / cfg.sr]
        prob = sess.run(None, {sess.get_inputs()[0].name: log_mel(x, cfg)[None]})[0][0]
        est = pick_hits(prob, cfg, classes, thr)
        s = transcription_scores(hits, est, classes)
        for c in classes:
            per_class[c]["p"].append(s[c].precision)
            per_class[c]["r"].append(s[c].recall)
            per_class[c]["f"].append(s[c].f_measure)
        if args.check_offset:
            offsets.append(estimate_offset(x, cfg.sr, hits, cfg))

    print(f"E-GMD {args.split} 分割 {len(tracks)} 本 × {args.seconds:.0f} 秒、しきい値 {thr}")
    print(f"{'種類':8s} {'P':>6s} {'R':>6s} {'F':>6s}")
    for c in classes:
        print(f"{c:8s} {np.mean(per_class[c]['p']):6.3f} {np.mean(per_class[c]['r']):6.3f} {np.mean(per_class[c]['f']):6.3f}")
    m0 = float(np.mean([np.mean(per_class[c]["f"]) for c in M0_CLASSES if c in per_class]))
    print(f"M0-1（kick/snare/hihat の F の平均）: {m0:.3f}  目安 0.85")
    if offsets:
        print(f"音声と MIDI のずれの見積もり: 中央値 {np.median(offsets) * 1000:.1f} ms（正なら音声が遅い）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
