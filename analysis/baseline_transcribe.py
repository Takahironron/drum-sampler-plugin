#!/usr/bin/env python3
"""学習モデル無しの採譜（基準値）。WAV を読んで打点 CSV を書く。

使い方::

    python baseline_transcribe.py ステム.wav 推定.csv [--threshold-lowmid 0.3 --threshold-high 0.06 --kick-ratio 8]

結果は `score_transcription.py` で正解と比べる。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drumeval import save_hits_csv  # noqa: E402
from drumeval.baseline import BaselineConfig, transcribe  # noqa: E402
from stft_distance import load_mono  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav")
    ap.add_argument("out_csv")
    ap.add_argument("--threshold-lowmid", type=float, default=BaselineConfig.threshold_lowmid, help="キック・スネアの山のしきい値")
    ap.add_argument("--threshold-high", type=float, default=BaselineConfig.threshold_high, help="ハイハットの山のしきい値")
    ap.add_argument("--kick-ratio", type=float, default=BaselineConfig.kick_energy_ratio, help="打点直後の 低／中 のエネルギー比がこれ以上なら kick")
    args = ap.parse_args(argv)

    x, sr = load_mono(args.wav, None)
    cfg = BaselineConfig(threshold_lowmid=args.threshold_lowmid, threshold_high=args.threshold_high, kick_energy_ratio=args.kick_ratio)
    hits = transcribe(x, sr, cfg)
    save_hits_csv(hits, args.out_csv)
    print(f"{len(hits)} 打を書き出した: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
