#!/usr/bin/env python3
"""採譜モデルの学習に使う「にじみ」（ドラム以外の楽器の音）を、自作の合成音で作る。権利の明確な音だけ（製品に入れるモデル用）。

使い方::

    python make_bleed_set.py data/bleed_synth [--minutes 60] [--seconds 30] [--sr 22050] [--seed 0]

`data/bleed_synth/bleed_0001.flac` … を書く（1 本 `--seconds` 秒、合わせて `--minutes` 分）。学習では::

    python train_transcriber.py --egmd <E-GMD のフォルダ> --bleed-dir data/bleed_synth --out models/crnn_egmd_v6.onnx

- 音はすべて `drumeval/bleed.py` が合成する（ベース、撥弦、鍵盤、パッド、リード、声の母音と子音）。音声の素材も、既存の曲の MIDI も使わない。
- v5 までは、にじみに区分 A（CD 曲）のステムを使ったので、モデルを製品に入れられなかった（R3）。
- `data/` は `.gitignore` 済み。作った音はコミットしない（スクリプトと乱数の種から作り直せる）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drumeval.bleed import render_bleed  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    import soundfile as sf

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--minutes", type=float, default=60.0, help="合わせて何分ぶん作るか")
    ap.add_argument("--seconds", type=float, default=30.0, help="1 本の長さ（秒）")
    ap.add_argument("--sr", type=int, default=22050, help="サンプリング周波数（学習の特徴量は 22.05 kHz）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_files = max(int(round(args.minutes * 60.0 / args.seconds)), 1)
    for i in range(n_files):
        rng = np.random.default_rng([args.seed, i])  # 1 本ごとに決まった乱数（途中からやり直しても同じ音）
        y = render_bleed(args.seconds, args.sr, rng)
        sf.write(out / f"bleed_{i + 1:04d}.flac", y, args.sr, subtype="PCM_16")
        if (i + 1) % 20 == 0 or i + 1 == n_files:
            print(f"  {i + 1} / {n_files}", flush=True)
    (out / "README.txt").write_text(
        "採譜モデルの学習の増強に使う「にじみ」（ドラム以外の楽器の音）。analysis/make_bleed_set.py が合成した音だけ"
        f"（seed {args.seed}、{n_files} 本 × {args.seconds:g} 秒、{args.sr} Hz）。音声の素材も既存の曲の MIDI も使っていない。\n", encoding="utf-8")
    print(f"書き出した: {out}（{n_files} 本、合わせて {n_files * args.seconds / 60:.0f} 分）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
