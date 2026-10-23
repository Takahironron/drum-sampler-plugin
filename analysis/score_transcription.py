#!/usr/bin/env python3
"""採譜の F 値を出す（M0-1）。

使い方::

    python score_transcription.py 正解.csv 推定.csv [--tolerance 0.05] [--json]

正解・推定は CSV（time_sec,class[,velocity]）か MIDI（GM ドラムマップ）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drumeval import COARSE_CLASSES, load_hits, m0_transcription_score, transcription_scores  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", help="正解（CSV か MIDI）")
    ap.add_argument("estimated", help="推定（CSV か MIDI）")
    ap.add_argument("--tolerance", type=float, default=0.05, help="時刻の許容誤差（秒）。既定 0.05")
    ap.add_argument("--json", action="store_true", help="JSON で出す")
    args = ap.parse_args(argv)

    ref = load_hits(args.reference)
    est = load_hits(args.estimated)
    scores = transcription_scores(ref, est, COARSE_CLASSES, args.tolerance)
    m0 = m0_transcription_score(ref, est, args.tolerance)

    if args.json:
        print(json.dumps({"classes": {c: s.__dict__ for c, s in scores.items()}, "m0_f": m0}, ensure_ascii=False, indent=2))
        return 0

    print(f"{'種類':8s} {'P':>6s} {'R':>6s} {'F':>6s} {'正解':>5s} {'推定':>5s}")
    for c, s in scores.items():
        print(f"{c:8s} {s.precision:6.3f} {s.recall:6.3f} {s.f_measure:6.3f} {s.n_ref:5d} {s.n_est:5d}")
    print(f"M0-1（kick/snare/hihat の F の平均）: {m0:.3f}  目安 0.85")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
