#!/usr/bin/env python3
"""プラグインのエンジン（C++ の `SampleEngine`）が、Python の参照実装（`drumeval.extract.render_midi`）と同じ音を出すかを確かめる。

キット（`kit.json`）と MIDI（打点の時刻・種類・強さ）から鳴る音が、解析（Python）と音源（C++）で同じであることが、
「解析の結果を聞いて確かめたとおりにプラグインで鳴る」ことの保証になる。CI（`.github/workflows/build.yml`）で毎回確かめる。

使い方::

    python engine_parity.py make  キットのフォルダ 作業フォルダ      # 打点のリスト（events.csv）と、Python で鳴らした音（python.wav）を書く
    KitRenderTest キットのフォルダ/kit.json 作業フォルダ/cpp.wav --events 作業フォルダ/events.csv
    python engine_parity.py check キットのフォルダ 作業フォルダ      # cpp.wav と python.wav を比べる（違えば終了コード 1）

- 打点のリストは、キットにある種類を使った乱数のパターン（強さ 1〜127、同時の打、16 分の連打、同じ位置の重なり）。
- C++ のエンジンは音を頭から鳴らす（立ち上がりは `lead_sec` 後）。プラグインはこの長さを遅れとしてホストに伝え、ホストが補正する。
  ここでは C++ の出力を同じ長さだけ前にずらして比べる。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drumeval import Hit  # noqa: E402
from drumeval.extract import SAMPLE_LEAD_SEC, Kit, load_kit, render_midi  # noqa: E402
from drumeval.labels import COARSE_TO_NOTE  # noqa: E402

TOLERANCE = 1e-4  # 1 サンプルごとの差の上限（振幅）。C++ は 32 bit float で足し合わせるので、丸めの分だけ違う


def make_events(kit: Kit, seconds: float = 8.0, seed: int = 0) -> list[tuple[int, int, int]]:
    """(サンプル位置, ノート番号, 強さ) の並び。プラグインに来るノートオンと同じ形。"""
    rng = np.random.default_rng(seed)
    sr = kit.sr
    step = int(round(0.125 * sr))  # 120 BPM の 16 分
    n_steps = int(seconds * sr) // step
    events: list[tuple[int, int, int]] = []
    for cls in kit.parts:
        note = COARSE_TO_NOTE[cls]
        p = {"kick": 0.3, "snare": 0.2, "hihat": 0.7, "tom": 0.15, "cymbal": 0.06}.get(cls, 0.2)
        for k in range(n_steps):
            if rng.random() < p:
                jitter = int(rng.integers(-40, 41))  # 格子から少し外す（サンプル単位の位置を試す）
                events.append((max(k * step + jitter, 0), note, int(rng.integers(1, 128))))
        events.append((0, note, 127))  # 先頭の打（頭の切れ方）と、強さの両端
        events.append((3 * step, note, 1))
        events.append((5 * step, note, 127))
        events.append((5 * step, note, 64))  # 同じ位置に同じ種類の打が重なる
        events.append((5 * step + 30, note, 90))  # フェードの途中で次の打が来る
    events.sort(key=lambda e: e[0])  # 同じ位置の打は、上で足した順のまま（C++ も同じ順に処理する）
    return events


def reference(kit: Kit, events: list[tuple[int, int, int]], seconds: float) -> np.ndarray:
    note_to_cls = {n: c for c, n in COARSE_TO_NOTE.items()}
    hits = [Hit(pos / kit.sr, note_to_cls[note], vel) for pos, note, vel in events]
    return render_midi(hits, kit, duration=seconds)


def cmd_make(kit_dir: Path, work: Path, seconds: float, seed: int) -> int:
    import soundfile as sf

    kit = load_kit(kit_dir)
    events = make_events(kit, seconds, seed)
    work.mkdir(parents=True, exist_ok=True)
    with open(work / "events.csv", "w", encoding="utf-8", newline="\n") as f:
        f.write("# sample,note,velocity\n")
        for pos, note, vel in events:
            f.write(f"{pos},{note},{vel}\n")
    longest = max(s.size for c in kit.parts for s, _, _ in kit.all_samples(c)) / kit.sr
    total = events[-1][0] / kit.sr + longest + 0.5
    y = reference(kit, events, total)
    sf.write(work / "python.wav", y.astype(np.float32), kit.sr, subtype="FLOAT")
    (work / "parity.json").write_text(json.dumps({"seconds": total, "events": len(events), "sr": kit.sr}), encoding="utf-8")
    print(f"書き出した: {work / 'events.csv'}（打 {len(events)}）、{work / 'python.wav'}（{total:.2f} 秒、ピーク {np.max(np.abs(y)):.3f}）")
    return 0


def compare(py: np.ndarray, cpp: np.ndarray, lead: int) -> dict:
    """C++ の出力を lead サンプルだけ前にずらして比べる。"""
    cpp = cpp[lead:]
    n = min(py.size, cpp.size)
    diff = np.abs(py[:n] - cpp[:n])
    tail = max(float(np.max(np.abs(py[n:]))) if py.size > n else 0.0, float(np.max(np.abs(cpp[n:]))) if cpp.size > n else 0.0)
    worst = int(np.argmax(diff)) if n else 0
    return {"n": n, "max_diff": float(diff.max()) if n else 0.0, "worst_sample": worst, "rms_diff": float(np.sqrt(np.mean(diff**2))) if n else 0.0,
            "peak": float(np.max(np.abs(py))) if py.size else 0.0, "tail": tail}


def cmd_check(kit_dir: Path, work: Path, cpp_wav: Path | None) -> int:
    import soundfile as sf

    meta = json.loads((kit_dir / "kit.json").read_text(encoding="utf-8"))
    py, sr = sf.read(work / "python.wav", dtype="float64")
    cpp, sr2 = sf.read(cpp_wav or work / "cpp.wav", dtype="float64", always_2d=True)
    if sr != sr2:
        print(f"サンプリング周波数が違う: Python {sr}、C++ {sr2}")
        return 1
    lead = int(round(float(meta.get("lead_sec", SAMPLE_LEAD_SEC)) * sr))
    r = compare(py, cpp[:, 0], lead)
    ok = r["max_diff"] <= TOLERANCE and r["tail"] <= TOLERANCE and r["peak"] > 0.01
    print(f"比べた長さ {r['n'] / sr:.2f} 秒、ピーク {r['peak']:.3f}、差の最大 {r['max_diff']:.2e}（{r['worst_sample'] / sr:.4f} 秒の位置）、"
          f"差の RMS {r['rms_diff']:.2e}、はみ出した部分の最大 {r['tail']:.2e}")
    print("一致" if ok else f"不一致（上限 {TOLERANCE:g}）")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("make", "check"))
    ap.add_argument("kit_dir", help="kit.json のあるフォルダ")
    ap.add_argument("work_dir", help="events.csv、python.wav、cpp.wav を置くフォルダ")
    ap.add_argument("--seconds", type=float, default=8.0, help="打点を置く長さ（make）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpp", default=None, help="C++ の出力（check。省略すると 作業フォルダ/cpp.wav）")
    args = ap.parse_args(argv)
    if args.command == "make":
        return cmd_make(Path(args.kit_dir), Path(args.work_dir), args.seconds, args.seed)
    return cmd_check(Path(args.kit_dir), Path(args.work_dir), Path(args.cpp) if args.cpp else None)


if __name__ == "__main__":
    raise SystemExit(main())
