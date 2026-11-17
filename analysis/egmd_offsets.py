#!/usr/bin/env python3
"""E-GMD の正解（MIDI）と音の時刻のずれを、ファイルごとに測って一覧にする。

E-GMD の配布物では、正解の MIDI と音がファイルごとに一定量ずれているものがある（2026-11-16、標本 67 本: 10 ms 以上が 34%、
30 ms 以上が 15%、50 ms 以上が 7%、範囲は −73〜+119 ms。同じキットでもファイルによって違う）。ずれたままだと、
採譜モデルは数フレームずれたラベルで学習し、採点（許容 ±50 ms）は正しい検出を誤りと数える。

使い方::

    python egmd_offsets.py D:\\data\\e-gmd-v1.0.0 [--workers 4] [--split all] [--max-tracks 0]
        E-GMD の全トラックのずれを測り、同じフォルダの label_offsets.csv に書く（midi_filename, offset_ms, sharpness, n_hits）。
        途中で止めても、もう一度走らせれば続きから。学習（train_transcriber.py）と採点（eval_egmd.py）は、この一覧があれば自動で使う。
    python egmd_offsets.py --stems data/egmd_stems
        run_eval.py の形のフォルダ（<曲ID>/drums.wav と hits.mid）の各曲に、ずれ（秒）を label_offset.txt として書く。
        run_eval.py は、正解の打点にこの値を足して使う。

- ずれ（offset_ms）は「音が MIDI より何 ms 遅れているか」。正解の打点に足すと音に合う。測り方は `drumeval/align.py`
  （音の立ち上がりの強さと打点の列の相互相関、±250 ms）。sharpness は山のはっきりさで、1.5 未満のものは使わない。
- 時間の目安: 音 1 分あたり 0.2〜0.3 秒（1 プロセス）。E-GMD 全体（444 時間）で、4 プロセスなら 30〜40 分 ＋ 読み込み。
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drumeval.align import MIN_SHARPNESS, label_offset  # noqa: E402

SEARCH_MS = 250.0  # `label_offset` がずれを探す範囲（±）。`max_lag_sec` の既定と同じ

SR = 22050  # ずれを測るときの sr（半分にしても結果の差は 0.6 ms 以内）


def measure(audio: str, midi: str) -> tuple[float, float, int]:
    """1 トラックのずれ（ms）、確かさ、打の数。読めなければ (0, 0, 0)。"""
    from drumeval.egmd import load_audio
    from drumeval.labels import load_hits

    try:
        x = load_audio(audio, SR)
        hits = load_hits(midi)
        off, sharp = label_offset(x, SR, hits)
    except Exception as e:  # noqa: BLE001  壊れたファイルで全体を止めない
        print(f"読めない: {audio}（{e}）", file=sys.stderr)
        return 0.0, 0.0, 0
    return off * 1000.0, sharp, len(hits)


def _measure_row(args: tuple[str, str, str]) -> tuple[str, float, float, int]:
    key, audio, midi = args
    return (key, *measure(audio, midi))


def summarize(offsets_ms: list[float], sharpness: list[float]) -> str:
    v = np.abs(np.asarray([o for o, s in zip(offsets_ms, sharpness) if s >= MIN_SHARPNESS]))
    if v.size == 0:
        return "測れたトラックが無い"
    weak = sum(1 for s in sharpness if s < MIN_SHARPNESS)
    signed = np.asarray([o for o, s in zip(offsets_ms, sharpness) if s >= MIN_SHARPNESS])
    big = signed[np.abs(signed) >= 10]
    edge = int(np.sum(v >= SEARCH_MS - 10))
    return (f"{v.size} 本: ずれの絶対値の中央 {np.median(v):.1f} ms、10 ms 以上 {np.mean(v >= 10) * 100:.0f}%、20 ms 以上 {np.mean(v >= 20) * 100:.0f}%、"
            f"30 ms 以上 {np.mean(v >= 30) * 100:.0f}%、50 ms 以上 {np.mean(v >= 50) * 100:.0f}%、最大 {v.max():.0f} ms"
            + (f"。山がはっきりせず使わないもの {weak} 本" if weak else "")
            + (f"\n10 ms 以上のうち、音が遅いもの（正）{np.mean(big > 0) * 100:.0f}%、早いもの（負）{np.mean(big < 0) * 100:.0f}%。100 ms 以上 {int(np.sum(v >= 100))} 本、"
               f"探す範囲（±{SEARCH_MS:.0f} ms）の端から 10 ms 以内 {edge} 本" + ("（本当のずれは範囲の外かもしれない）" if edge else "") if big.size else ""))


def sequence_of(key: str) -> str:
    """E-GMD の midi_filename から、演奏（元の MIDI。キット違いの録音で共通）の名前を取る: 末尾の「_キット番号」を外す。"""
    m = re.match(r"^(.*)_(\d+)\.midi?$", key)
    return m.group(1) if m else key


def structure(rows: list[tuple[str, str, float]]) -> list[str]:
    """ずれが、演奏（元の MIDI）とキットのどちらで決まっているかの要約。`rows` は (演奏, キット, ずれ ms)。

    組（同じ演奏、または同じキット。3 本以上あるもの）で分けたときに、ずれのばらつき（平均からの差の二乗和）のうち
    組の平均で説明できる割合と、組の中での散らばり（組の中央値からの差の絶対値の中央値。組をまたいだ中央値）を出す。
    """
    lines: list[str] = []
    if len(rows) < 30:
        return lines
    for label, idx in (("演奏（元の MIDI。キット違いの録音で共通）", 0), ("キット", 1)):
        groups: dict[str, list[float]] = {}
        for r in rows:
            groups.setdefault(r[idx], []).append(r[2])
        big = [np.asarray(v) for v in groups.values() if len(v) >= 3]
        if len(big) < 2:
            continue
        inside = np.concatenate(big)
        total = float(np.sum((inside - inside.mean()) ** 2)) + 1e-12
        within = float(sum(np.sum((v - v.mean()) ** 2) for v in big))
        spread = float(np.median([np.median(np.abs(v - np.median(v))) for v in big]))
        lines.append(f"{label}で分けると、ずれのばらつきの {100.0 * (1.0 - within / total):.0f}% を説明する（{len(big)} 組、{inside.size} 本）。"
                     f"組の中での散らばり {spread:.1f} ms")
    return lines


def run_dataset(root: Path, workers: int, split: str, max_tracks: int) -> int:
    from concurrent.futures import ProcessPoolExecutor

    from drumeval.egmd import OFFSETS_CSV, find_index_csv, read_index, read_offsets

    base = find_index_csv(root).parent
    out = base / OFFSETS_CSV
    done = read_offsets(base)
    tracks = [t for t in read_index(root, use_offsets=False) if split == "all" or t.split == split]
    if max_tracks:
        tracks = tracks[:max_tracks]
    todo = [(t.key, str(t.audio), str(t.midi)) for t in tracks if t.key not in done]
    print(f"{len(tracks)} 本のうち、測り済み {len(tracks) - len(todo)} 本、これから {len(todo)} 本（{workers} プロセス）。書き先: {out}", flush=True)
    new_file = not out.exists()
    pool = ProcessPoolExecutor(workers) if workers > 1 and todo else None
    try:
        with open(out, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["midi_filename", "offset_ms", "sharpness", "n_hits"])
            results = pool.map(_measure_row, todo, chunksize=8) if pool else map(_measure_row, todo)
            for i, (key, off, sharp, n) in enumerate(results, 1):
                w.writerow([key, f"{off:.2f}", f"{sharp:.2f}", n])
                if i % 200 == 0 or i == len(todo):
                    f.flush()  # 途中で止めても、ここまでは残る
                    print(f"  {i} / {len(todo)}", flush=True)
    finally:
        if pool:
            pool.shutdown(wait=False, cancel_futures=True)
    allv = read_offsets(base)
    keys = [t.key for t in tracks if t.key in allv]
    print(summarize([allv[k][0] * 1000 for k in keys], [allv[k][1] for k in keys]))
    for line in structure([(sequence_of(t.key), t.kit or "?", allv[t.key][0] * 1000) for t in tracks if t.key in allv and allv[t.key][1] >= MIN_SHARPNESS]):
        print(line)
    return 0


def run_stems(stems: Path) -> int:
    from drumeval import load_hits
    from stft_distance import load_mono

    from drumeval.stems import find_stem

    offs, sharps = [], []
    for d in sorted(p for p in stems.iterdir() if p.is_dir()):
        truth = next((p for p in (d / "hits.csv", d / "hits.mid", d / "hits.midi") if p.exists()), None)
        audio = find_stem(d)
        if truth is None or audio is None:
            continue
        x, sr = load_mono(str(audio), None)
        off, sharp = label_offset(x, sr, load_hits(truth))
        use = off if sharp >= MIN_SHARPNESS else 0.0
        (d / "label_offset.txt").write_text(f"{use:.4f}\n", encoding="utf-8")
        offs.append(off * 1000)
        sharps.append(sharp)
        print(f"  {d.name}: {off * 1000:+.1f} ms（確かさ {sharp:.1f}）" + ("" if sharp >= MIN_SHARPNESS else " → 使わない（0 を書く）"))
    print(summarize(offs, sharps))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("egmd", nargs="?", help="E-GMD のフォルダ（e-gmd-v1.0.0.csv のあるところ）")
    ap.add_argument("--stems", default=None, help="run_eval.py の形のフォルダ。各曲に label_offset.txt を書く")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--split", default="all", help="train / validation / test / all")
    ap.add_argument("--max-tracks", type=int, default=0, help="先頭からこの本数だけ測る（0 で全部。試すとき用）")
    args = ap.parse_args(argv)
    if args.stems:
        return run_stems(Path(args.stems))
    if not args.egmd:
        ap.error("E-GMD のフォルダか --stems が要る")
    return run_dataset(Path(args.egmd), args.workers, args.split, args.max_tracks)


if __name__ == "__main__":
    raise SystemExit(main())
