#!/usr/bin/env python3
"""評価用ステムの集まり（`data/stems/<曲ID>/`）に対して、M0 の3指標をまとめて出す（回帰テスト）。

使い方::

    python run_eval.py data/stems [--out docs/phase0/結果_YYYY-MM-DD.md] [--est-dir 推定の置き場] [--max-seconds 60]

各曲のフォルダ:
    drums.wav | .flac    分離済みのドラムのステム（必須）
    hits.csv | hits.mid  打点の正解（あれば M0-1 を出す）

採譜の推定は、`--est-dir` に `<曲ID>.csv` があればそれを使い、無ければ基準値（`baseline_transcribe`）で作る。
出力は Markdown の表（曲ごとの F 値、切り出せた打の数と層、再現との STFT 距離）。
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drumeval import load_hits, multi_resolution_stft_distance, transcription_scores  # noqa: E402
from drumeval.baseline import transcribe  # noqa: E402
from drumeval.extract import build_kit, cut_segments, estimate_velocities, render_kit  # noqa: E402
from drumeval.labels import M0_CLASSES  # noqa: E402
from stft_distance import load_mono  # noqa: E402


def evaluate_song(song_dir: Path, est_csv: Path | None, max_seconds: float) -> dict:
    wav = next(p for p in (song_dir / "drums.wav", song_dir / "drums.flac") if p.exists())
    x, sr = load_mono(str(wav), None)
    if max_seconds and x.size > max_seconds * sr:
        x = x[: int(max_seconds * sr)]
    row: dict = {"id": song_dir.name, "seconds": x.size / sr}

    if est_csv and est_csv.exists():
        est = load_hits(est_csv)
        row["est_source"] = est_csv.name
    else:
        est = transcribe(x, sr)
        row["est_source"] = "baseline"
    est = [h for h in est if h.time < x.size / sr]

    truth_path = next((p for p in (song_dir / "hits.csv", song_dir / "hits.mid", song_dir / "hits.midi") if p.exists()), None)
    if truth_path:
        ref = [h for h in load_hits(truth_path) if h.time < x.size / sr]
        scores = transcription_scores(ref, est)
        for c in M0_CLASSES:
            row[f"f_{c}"] = scores[c].f_measure
        row["m0_1"] = float(np.mean([scores[c].f_measure for c in M0_CLASSES]))

    hits = estimate_velocities(est, None, x, sr) if any(h.velocity is None for h in est) else est
    segs = cut_segments(x, sr, hits)
    kit = build_kit(segs, sr)
    row["n_hits"] = len(hits)
    row["n_isolated"] = len(segs)
    row["layers"] = ", ".join(f"{c}:{len(ls)}/{kit.n_samples(c)}" for c, ls in kit.parts.items())
    if kit.parts:
        y = render_kit(hits, kit, duration=x.size / sr)
        row["stft_rerender"] = multi_resolution_stft_distance(x, y)["total"]
    return row


def to_markdown(rows: list[dict], title: str) -> str:
    lines = [f"# {title}", "", "| 曲ID | 秒 | 推定 | F kick | F snare | F hihat | M0-1 | 打 | 重ならない打 | 層/音（種類ごと） | STFT 距離（元 vs 再現） |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    def f(v):
        return "" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    for r in rows:
        lines.append("| " + " | ".join(f(r.get(k)) for k in ("id", "seconds", "est_source", "f_kick", "f_snare", "f_hihat", "m0_1", "n_hits", "n_isolated", "layers", "stft_rerender")) + " |")
    m0 = [r["m0_1"] for r in rows if "m0_1" in r]
    lines += ["", f"- 正解付きの曲: {len(m0)} / {len(rows)}"]
    if m0:
        lines.append(f"- M0-1（kick/snare/hihat の F の平均）の平均: {np.mean(m0):.3f}（目安 0.85）")
    d = [r["stft_rerender"] for r in rows if "stft_rerender" in r]
    if d:
        lines.append(f"- 再現との STFT 距離の平均: {np.mean(d):.3f}（M0-2 の聞き比べの参考。小さいほど近い）")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems_dir")
    ap.add_argument("--out", default=None, help="Markdown の出力先。省略時は標準出力")
    ap.add_argument("--est-dir", default=None, help="採譜の推定 CSV（<曲ID>.csv）の置き場")
    ap.add_argument("--max-seconds", type=float, default=60.0, help="曲の先頭から使う長さ（0 で全部）")
    args = ap.parse_args(argv)

    stems = sorted(p for p in Path(args.stems_dir).iterdir() if (p / "drums.wav").exists() or (p / "drums.flac").exists())
    if not stems:
        print("drums.wav のあるフォルダが見つからない:", args.stems_dir, file=sys.stderr)
        return 1
    rows = []
    for d in stems:
        est = Path(args.est_dir) / f"{d.name}.csv" if args.est_dir else None
        print(f"{d.name} ...", file=sys.stderr)
        rows.append(evaluate_song(d, est, args.max_seconds))
    md = to_markdown(rows, f"評価の結果（{date.today().isoformat()}）")
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print("書き出した:", args.out, file=sys.stderr)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
