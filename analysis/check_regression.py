#!/usr/bin/env python3
"""解析の回帰テスト: `run_eval.py --json` の要約を、リポジトリに置いた基準（`regression_baseline.json`）と比べる。

解析（パート分け、打点の合わせ直し、切り出し、キット、鳴らし方）を変えたときに、固定の評価セットでの数字が悪くなっていないかを確かめる。
評価セットは、クラウドでも取れる CC BY 4.0 の標本（`docs/phase0/評価用ステム一覧.md` の「クラウドでも使える標本」）。
CI（`.github/workflows/analysis-regression.yml`）で、解析の本体を変えたときに走る。区分 A（CD 曲）は Takahiro の PC にしか無いので、
ここでは測れない（聞き比べで確かめる）。

使い方::

    python analysis/run_eval.py data/stemgmd_stems --parts nmfd-oracle --json data/results/regression_stemgmd.json --out data/results/regression_stemgmd.md
    python analysis/check_regression.py data/results/regression_stemgmd.json stemgmd            # 基準より悪ければ終了コード 1
    python analysis/check_regression.py data/results/regression_stemgmd.json stemgmd --update   # 今の値を基準にする（良くなったとき）

- 基準の指標は、どれも小さいほど良い（距離、位置がずれる打の割合）。「今の値 > 基準 + 許容」なら悪化とみなす。
- 許容は、計算機の違い（BLAS の丸め）による揺れより大きく、意味のある悪化より小さく取る。
- 基準より「許容」以上良くなったら、その旨を出す。直した結果なら `--update` で基準を更新してコミットする。
- 曲の数が基準と違うとき（評価セットを取り損ねた）は失敗にする。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_BASELINE = Path(__file__).resolve().parent / "regression_baseline.json"

# 基準に無い指標を --update で足すときの許容（指標の名前の頭で決める）
DEFAULT_TOLERANCES = (("midi_offsets_hihat", 0.04), ("midi_offsets_", 0.02), ("part_dist_midi_hihat", 0.05), ("part_dist_", 0.03), ("stft_", 0.03))


def default_tolerance(metric: str) -> float:
    for prefix, tol in DEFAULT_TOLERANCES:
        if metric.startswith(prefix):
            return tol
    return 0.03


def compare(current: dict, baseline_set: dict) -> tuple[bool, list[str]]:
    """返り値: (合格か, 表示する行)。"""
    lines: list[str] = []
    ok = True
    if current.get("n_songs") != baseline_set.get("n_songs"):
        ok = False
        lines.append(f"曲の数が基準と違う: 今 {current.get('n_songs')}、基準 {baseline_set.get('n_songs')}（評価セットを取り直す）")
    means = current.get("means", {})
    lines.append(f"{'指標':28s} {'基準':>8s} {'今':>8s} {'差':>8s} {'許容':>6s}")
    for metric, spec in baseline_set.get("metrics", {}).items():
        base, tol = float(spec["value"]), float(spec["tolerance"])
        if metric not in means:
            ok = False
            lines.append(f"{metric:28s} {base:8.3f} {'無い':>8s}  ← 悪化（値が出ていない）")
            continue
        now = float(means[metric])
        diff = now - base
        note = ""
        if diff > tol:
            ok = False
            note = "  ← 悪化"
            worst = sorted(((v.get(metric, 0.0), song) for song, v in current.get("per_song", {}).items() if metric in v), reverse=True)[:3]
            if worst:
                note += "（値の大きい曲: " + "、".join(f"{song} {v:.3f}" for v, song in worst) + "）"
        elif diff < -tol:
            note = "  ← 基準より良い（直した結果なら --update で基準を更新する）"
        lines.append(f"{metric:28s} {base:8.3f} {now:8.3f} {diff:+8.3f} {tol:6.3f}{note}")
    return ok, lines


def update(current: dict, baseline: dict, set_name: str, command: str | None) -> None:
    sets = baseline.setdefault("sets", {})
    old = sets.get(set_name, {})
    metrics = old.get("metrics", {})
    means = current.get("means", {})
    for metric in list(metrics):
        if metric in means:
            metrics[metric] = {"value": round(float(means[metric]), 4), "tolerance": metrics[metric]["tolerance"]}
    sets[set_name] = {"command": command or old.get("command", ""), "n_songs": current.get("n_songs"), "metrics": metrics}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary", help="run_eval.py --json の出力")
    ap.add_argument("set_name", help="基準の中の評価セットの名前（例: stemgmd、egmd）")
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    ap.add_argument("--update", action="store_true", help="今の値を基準に書き込む（許容はそのまま）")
    ap.add_argument("--add", default="", help="--update のときに基準へ足す指標（コンマ区切り。例: stft_midi,part_dist_midi_kick）")
    ap.add_argument("--command", default=None, help="--update のときに基準へ書いておく、測り方のコマンド")
    args = ap.parse_args(argv)

    current = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    path = Path(args.baseline)
    baseline = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"sets": {}}
    if args.update:
        sets = baseline.setdefault("sets", {})
        entry = sets.setdefault(args.set_name, {"command": "", "n_songs": current.get("n_songs"), "metrics": {}})
        for metric in [m.strip() for m in args.add.split(",") if m.strip()]:
            if metric not in current.get("means", {}):
                print(f"要約に無い指標: {metric}", file=sys.stderr)
                return 1
            entry["metrics"].setdefault(metric, {"value": 0.0, "tolerance": default_tolerance(metric)})
        update(current, baseline, args.set_name, args.command)
        path.write_text(json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"基準を更新した: {path}（{args.set_name}、指標 {len(baseline['sets'][args.set_name]['metrics'])} 個）")
        return 0
    if args.set_name not in baseline.get("sets", {}):
        print(f"基準に評価セット {args.set_name} が無い（{path}）。--update で作る", file=sys.stderr)
        return 1
    ok, lines = compare(current, baseline["sets"][args.set_name])
    print(f"回帰テスト: {args.set_name}（{current.get('n_songs')} 曲）")
    for line in lines:
        print(line)
    print("合格（基準より悪くなっていない）" if ok else "不合格（基準より悪い指標がある）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
