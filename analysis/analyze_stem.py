#!/usr/bin/env python3
"""ドラムのステム 1 本を解析して、キットと MIDI を作る（解析の参照実装。計画書のモジュール `Analyzer`）。

使い方::

    python analyze_stem.py ドラム.wav 出力フォルダ --model models/crnn_egmd_v5.onnx [--start 秒] [--seconds 秒] [--kit-size 24] [--preview]

出力フォルダにできるもの:
    kit.json と <種類>/layer1_rrN.wav   キット。プラグインの「kit.json を読む」で開ける
    hits.mid                             打点（GM ドラムマップ、チャンネル 10、120 BPM。時刻は元のステムの先頭から）
    analysis.json                        打の数、キットの音の数、強さと音量の対応、所要時間
    preview_midi.wav（--preview）        このキットを hits.mid だけで鳴らした音（プラグインで鳴る音の予想）
    original_cut.wav（--preview）        解析した区間の元の音（聞き比べ用）

`hits.mid` を DAW に置き、プラグインにキットを読み込めば、元のステムに近い音が鳴る。強さ 1〜127 は、種類ごとの打の音量の範囲
（`kit.json` の `level_range_db`）に dB で線形に対応している。

- `--hits` に打点（CSV か MIDI）を渡すと、採譜モデルの代わりにそれを使う（正解の打点で試すとき）。
- 評価（正解との比較、方式ごとの距離、聞き比べ用の WAV 一式）は `run_eval.py`。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drumeval import Hit, load_hits, multi_resolution_stft_distance  # noqa: E402
from drumeval.analyzer import analyze, preview, save_analysis  # noqa: E402
from stft_distance import load_mono  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", help="ドラムのステム")
    ap.add_argument("out_dir", help="キットと MIDI を書くフォルダ")
    ap.add_argument("--model", default=None, help="ONNX の採譜モデル（--hits を渡さないときは必須）")
    ap.add_argument("--hits", default=None, help="打点（CSV か MIDI）。渡すと採譜モデルの代わりに使う。時刻はステムの先頭から")
    ap.add_argument("--threshold", type=float, default=None, help="採譜のしきい値（省略するとモデルの設定）")
    ap.add_argument("--start", type=float, default=0.0, help="解析を始める位置（秒）")
    ap.add_argument("--seconds", type=float, default=0.0, help="解析する長さ（秒。0 で最後まで）")
    ap.add_argument("--kit-size", type=int, default=24, help="種類ごとに残す代表の音の数")
    ap.add_argument("--preview", action="store_true", help="MIDI だけで鳴らした音（preview_midi.wav）と、元の音（original_cut.wav）も書く")
    args = ap.parse_args(argv)
    if not args.model and not args.hits:
        ap.error("--model（採譜モデル）か --hits（打点）のどちらかが要る")

    x, sr = load_mono(args.wav, None)
    x = x[int(args.start * sr) :]
    if args.seconds > 0:
        x = x[: int(args.seconds * sr)]
    hits = None
    if args.hits:
        dur = x.size / sr
        hits = [Hit(h.time - args.start, h.cls, h.velocity) for h in load_hits(args.hits) if 0.0 <= h.time - args.start < dur]
    a = analyze(x, sr, model=args.model, hits=hits, threshold=args.threshold, kit_size=args.kit_size)
    meta = save_analysis(a, args.out_dir, offset_sec=args.start)
    total = sum(a.timings.values())
    print(f"解析した長さ {a.duration:.1f} 秒、所要 {total:.1f} 秒（" + "、".join(f"{k} {v:.1f}" for k, v in a.timings.items()) + "）")
    print("打の数: " + " ".join(f"{c} {n}" for c, n in meta["hits"].items()) + " ／ キットの音の数: " + " ".join(f"{c} {n}" for c, n in meta["kit"].items()))
    if args.preview:
        import soundfile as sf

        y = preview(a)
        scale = (np.sqrt(np.mean(x**2)) + 1e-9) / (np.sqrt(np.mean(y**2)) + 1e-9)  # 聞き比べやすいように、元と同じ音量にそろえる
        y = y * scale
        peak = max(float(np.max(np.abs(y))), 1e-9)
        out = Path(args.out_dir)
        sf.write(out / "preview_midi.wav", (y / max(peak, 1.0)).astype(np.float32), sr)
        sf.write(out / "original_cut.wav", x.astype(np.float32), sr)
        d = multi_resolution_stft_distance(x, y)["total"]
        print(f"MIDI だけで鳴らした音と元の距離（多重解像度 STFT）: {d:.3f}。preview_midi.wav と original_cut.wav を聞き比べる")
    print(f"書き出した: {args.out_dir}（kit.json、hits.mid、analysis.json）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
