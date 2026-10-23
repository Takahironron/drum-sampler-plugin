#!/usr/bin/env python3
"""多重解像度 STFT 距離を出す（M0-3、切り出しの信頼度）。

使い方::

    python stft_distance.py 元のステム.wav 再現.wav [--sr 44100] [--json]

ステレオは左右を平均してモノラルにする。サンプリング周波数が違えば `--sr` に合わせて線形補間する。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drumeval import multi_resolution_stft_distance  # noqa: E402


def load_mono(path: str, target_sr: int | None) -> tuple[np.ndarray, int]:
    import soundfile as sf

    x, sr = sf.read(path, dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if target_sr is not None and sr != target_sr:
        n_new = int(round(x.size * target_sr / sr))
        x = np.interp(np.linspace(0, x.size - 1, n_new), np.arange(x.size), x)
        sr = target_sr
    return x, sr


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", help="基準（元のステム）")
    ap.add_argument("estimated", help="比較（再現）")
    ap.add_argument("--sr", type=int, default=None, help="揃えるサンプリング周波数。省略時は基準に合わせる")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    x, sr = load_mono(args.reference, args.sr)
    y, _ = load_mono(args.estimated, sr)
    d = multi_resolution_stft_distance(x, y)
    if args.json:
        print(json.dumps(d, indent=2))
    else:
        print(f"total={d['total']:.4f}  spectral_convergence={d['spectral_convergence']:.4f}  log_magnitude={d['log_magnitude']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
