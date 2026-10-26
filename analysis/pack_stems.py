#!/usr/bin/env python3
"""分離したステムを、受け渡し用に小さくする（モノラル 44.1 kHz の FLAC、先頭から指定秒数）。

使い方::

    python pack_stems.py 入力フォルダ 出力フォルダ [--seconds 90] [--start 0]

入力フォルダの WAV／FLAC をファイル名順に a01, a02, ... とし、`出力フォルダ/a01/drums.flac` のように書く。
入力のファイル名を `a03_xxx.wav` のように曲ID で始めておけば、その曲ID を使う。
`--start` で切り出しの開始位置（秒）を指定できる（ドラムが鳴っていない前奏を飛ばす）。
曲ごとに開始位置を変えるなら、ファイル名の末尾に `@45` のように秒を付ける（例: `a04_xxx@45.wav`）。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--sr", type=int, default=44100)
    args = ap.parse_args(argv)

    import soundfile as sf

    files = sorted(p for p in Path(args.src).iterdir() if p.suffix.lower() in (".wav", ".flac", ".aif", ".aiff"))
    if not files:
        print("音声ファイルが無い:", args.src, file=sys.stderr)
        return 1
    for i, f in enumerate(files, 1):
        m = re.match(r"^([a-z]\d\d)", f.stem)
        song_id = m.group(1) if m else f"a{i:02d}"
        m2 = re.search(r"@(\d+(?:\.\d+)?)$", f.stem)
        start = float(m2.group(1)) if m2 else args.start
        x, sr = sf.read(str(f), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        x = x[int(start * sr) : int((start + args.seconds) * sr)]
        if sr != args.sr:
            n_new = int(round(x.size * args.sr / sr))
            x = np.interp(np.linspace(0, x.size - 1, n_new), np.arange(x.size), x).astype(np.float32)
            sr = args.sr
        out = Path(args.dst) / song_id
        out.mkdir(parents=True, exist_ok=True)
        sf.write(out / "drums.flac", x, sr, subtype="PCM_16")
        print(f"{song_id}  {f.name}  {start:.0f}s から {x.size / sr:.0f}s  → {out / 'drums.flac'}  ({(out / 'drums.flac').stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
