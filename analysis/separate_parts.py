#!/usr/bin/env python3
"""ドラムのステムを NMFD でパート（キック／スネア／ハイハット／シンバル）に分ける（計画書 3.3 の (2)）。

使い方::

    python separate_parts.py ドラム.wav 打点.csv 出力フォルダ [--reference-dir パート別ステムのフォルダ] [--no-residual]

- 打点.csv は採譜の結果（`transcribe_onnx.py`）でも正解でもよい。活性の初期値になる。
- `--reference-dir` に分離ツールのパート別ステム（キック.wav など）があれば、SNR（dB）を出す。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drumeval import load_hits  # noqa: E402
from drumeval.nmfd import NMFDConfig, separate, snr_db  # noqa: E402
from run_eval import find_part_stems  # noqa: E402
from stft_distance import load_mono  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav")
    ap.add_argument("hits")
    ap.add_argument("out_dir")
    ap.add_argument("--reference-dir", default=None)
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--iterations", type=int, default=40)
    args = ap.parse_args(argv)

    import soundfile as sf

    x, sr = load_mono(args.wav, None)
    hits = load_hits(args.hits)
    cfg = NMFDConfig(n_fft=2048 if sr >= 40000 else 1024, hop=512 if sr >= 40000 else 256, residual=not args.no_residual, iterations=args.iterations)
    sep = separate(x, sr, hits, cfg)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = {"kick": "キック", "snare": "スネア", "hihat": "ハイハット", "cymbal": "シンバル", "residual": "残差"}
    for c, y in sep.items():
        sf.write(out / f"{names.get(c, c)}.wav", y.astype(np.float32), sr)
    print(f"書き出した: {out}（{', '.join(sep)}）")
    if args.reference_dir:
        refs = find_part_stems(Path(args.reference_dir))
        for c in sep:
            if c in refs:
                r, _ = load_mono(str(refs[c]), sr)
                print(f"  {c:7s} SNR {snr_db(r[: x.size], sep[c]):+.1f} dB（正解の代わり: {refs[c].name}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
