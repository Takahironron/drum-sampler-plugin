#!/usr/bin/env python3
"""合成音からデモ用のキット（kit.json と WAV）を作る。プラグインと C++ のエンジンの確認用。

使い方::

    python make_demo_kit.py 出力フォルダ [--sr 44100]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drumeval import Hit  # noqa: E402
from drumeval.extract import build_kit_by_features, cut_segments_from_parts, save_kit  # noqa: E402
from drumeval.labels import to_coarse  # noqa: E402
from drumeval.synth import random_pattern, render  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    hits: list[Hit] = []
    for k in range(4):  # 4 つの区間で音色を少しずつ変える
        seg = random_pattern(rng, 12.0)
        hits += [Hit(h.time + 12.5 * k, h.cls, h.velocity) for h in seg]
    dur = max(h.time for h in hits) + 1.5
    parts = {}
    for cls in ("kick", "snare", "hihat", "tom", "cymbal"):
        own = [h for h in hits if to_coarse(h.cls) == cls]
        if own:
            parts[cls] = (render(own, sr=args.sr, duration=dur, variation=0.6, seed=args.seed + 1), args.sr)
    segs = cut_segments_from_parts(parts, hits)
    kit = build_kit_by_features(segs, args.sr, n_per_class=16)
    save_kit(kit, args.out_dir)
    print("書き出した:", Path(args.out_dir) / "kit.json", " ".join(f"{c}:{kit.n_samples(c)}" for c in kit.parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
