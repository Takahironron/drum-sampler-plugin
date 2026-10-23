#!/usr/bin/env python3
"""乾いた音と目標（ステム）から、固定の形のエフェクトチェーンを当てはめる（M0-3）。

使い方::

    python fit_fx.py 乾いた音.wav 目標.wav [--out params.json] [--render 当てはめ後.wav] [--seconds 8]

2つの WAV は同じ打点で鳴らしたもの（例: `extract_kit.py --render` の再現と元のステム、
または内蔵ライブラリの音で鳴らしたものと元のステム）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drumeval.fx import apply_chain, fit_chain  # noqa: E402
from stft_distance import load_mono  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dry")
    ap.add_argument("wet")
    ap.add_argument("--out", default=None, help="パラメータの JSON")
    ap.add_argument("--render", default=None, help="当てはめ後の WAV")
    ap.add_argument("--seconds", type=float, default=8.0, help="当てはめに使う長さ（秒）")
    ap.add_argument("--maxiter", type=int, default=4)
    args = ap.parse_args(argv)

    dry, sr = load_mono(args.dry, None)
    wet, _ = load_mono(args.wet, sr)
    params, log = fit_chain(dry, wet, sr, max_seconds=args.seconds, maxiter=args.maxiter)

    print(f"STFT 距離  当てはめ前 {log['distance_before']:.4f} → 初期値 {log['distance_init']:.4f} → 当てはめ後 {log['distance_after']:.4f}（評価 {log['n_eval']} 回）")
    for k, v in params.to_dict().items():
        print(f"  {k:14s} {v:8.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps({"params": params.to_dict(), "log": log}, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.render:
        import soundfile as sf

        sf.write(args.render, apply_chain(dry, sr, params).astype(np.float32), sr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
