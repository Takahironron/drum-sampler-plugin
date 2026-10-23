#!/usr/bin/env python3
"""ONNX の採譜モデルで WAV から打点 CSV を作る（ONNX Runtime）。

使い方::

    python transcribe_onnx.py models/crnn_synth.onnx ステム.wav 推定.csv [--threshold 0.5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drumeval import save_hits_csv  # noqa: E402
from drumeval.infer import transcribe_onnx  # noqa: E402
from stft_distance import load_mono  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("wav")
    ap.add_argument("out_csv")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args(argv)
    x, sr = load_mono(args.wav, None)
    hits = transcribe_onnx(x, sr, args.model, args.threshold)
    save_hits_csv(hits, args.out_csv)
    print(f"{len(hits)} 打を書き出した: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
