#!/usr/bin/env python3
"""ステムと打点ラベルから、重ならない打を切り出して強さ別の層のキットを作る（M0-2 の材料）。

使い方::

    python extract_kit.py ステム.wav 打点.csv 出力フォルダ [--layers 3] [--rr 3] [--render 再現.wav]

- 打点.csv は採譜の結果でも正解でもよい。
- `--render` を付けると、切り出したキットで同じ打点を鳴らし直した WAV を書き、元のステムとの STFT 距離を出す
  （元のステム／再現 の A/B 聞き比べ用）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drumeval import load_hits, multi_resolution_stft_distance  # noqa: E402
from drumeval.extract import build_kit, cut_segments, estimate_velocities, render_kit, save_kit  # noqa: E402
from stft_distance import load_mono  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav")
    ap.add_argument("hits")
    ap.add_argument("out_dir")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--rr", type=int, default=3, help="層ごとのラウンドロビンの数")
    ap.add_argument("--render", default=None, help="キットで鳴らし直した WAV の出力先")
    args = ap.parse_args(argv)

    x, sr = load_mono(args.wav, None)
    hits = load_hits(args.hits)
    if any(h.velocity is None for h in hits):
        hits = estimate_velocities(hits, None, x, sr)
    segs = cut_segments(x, sr, hits)
    kit = build_kit(segs, sr, n_layers=args.layers, round_robin=args.rr)
    save_kit(kit, args.out_dir)

    print(f"打点 {len(hits)}、重ならない打 {len(segs)}")
    for cls, layers in kit.parts.items():
        print(f"  {cls:7s} 層 {len(layers)}  音 {kit.n_samples(cls)}")
    missing = sorted({h.cls for h in hits} - set(kit.parts))
    if missing:
        print("  切り出せなかった種類:", ", ".join(missing))

    if args.render:
        import soundfile as sf

        y = render_kit(hits, kit, duration=x.size / sr)
        sf.write(args.render, y.astype(np.float32), sr)
        d = multi_resolution_stft_distance(x, y)
        print(f"STFT 距離（元 vs 再現）: total={d['total']:.4f} sc={d['spectral_convergence']:.4f} logmag={d['log_magnitude']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
