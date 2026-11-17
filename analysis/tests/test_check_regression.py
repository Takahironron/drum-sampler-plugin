"""回帰テストの比べ方（check_regression.py）と、run_eval.py の要約（--json）を確かめる。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_regression  # noqa: E402
import run_eval  # noqa: E402


def _summary(stft_midi: float, kick_offsets: float, n_songs: int = 2) -> dict:
    return {"n_songs": n_songs, "songs": ["s1", "s2"][:n_songs],
            "means": {"stft_midi": stft_midi, "midi_offsets_kick": kick_offsets, "stft_self": 0.08},
            "per_song": {"s1": {"stft_midi": stft_midi + 0.1, "midi_offsets_kick": kick_offsets},
                         "s2": {"stft_midi": stft_midi - 0.1, "midi_offsets_kick": kick_offsets}}}


def test_update_then_compare(tmp_path, capsys):
    base = tmp_path / "baseline.json"
    cur = tmp_path / "now.json"

    def run(summary: dict, *extra: str) -> int:
        cur.write_text(json.dumps(summary), encoding="utf-8")
        code = check_regression.main([str(cur), "demo", "--baseline", str(base), *extra])
        return code

    assert run(_summary(1.00, 0.01)) == 1  # 基準が無い
    assert "--update で作る" in capsys.readouterr().err
    assert run(_summary(1.00, 0.01), "--update", "--add", "stft_midi,midi_offsets_kick", "--command", "run_eval ...") == 0
    data = json.loads(base.read_text(encoding="utf-8"))
    assert data["sets"]["demo"]["n_songs"] == 2 and data["sets"]["demo"]["command"] == "run_eval ..."
    assert data["sets"]["demo"]["metrics"] == {"stft_midi": {"value": 1.0, "tolerance": 0.03}, "midi_offsets_kick": {"value": 0.01, "tolerance": 0.02}}
    capsys.readouterr()

    assert run(_summary(1.02, 0.02)) == 0  # 許容の中
    assert "合格" in capsys.readouterr().out
    assert run(_summary(1.05, 0.01)) == 1  # 距離が 0.05 悪い
    out = capsys.readouterr().out
    assert "悪化" in out and "s1 1.150" in out and "不合格" in out  # 値の大きい曲も出す
    assert run(_summary(1.00, 0.10)) == 1  # キックの 1 割がずれる（2026-11-17 に直した不具合が戻ったとき）
    capsys.readouterr()
    assert run(_summary(0.90, 0.01)) == 0  # 良くなった分は合格。基準の更新を促す
    assert "基準より良い" in capsys.readouterr().out
    assert run(_summary(1.00, 0.01, n_songs=1)) == 1  # 評価セットが欠けている
    assert "曲の数が基準と違う" in capsys.readouterr().out
    missing = _summary(1.00, 0.01)
    del missing["means"]["midi_offsets_kick"]
    assert run(missing) == 1  # 基準にある指標が出ていない
    capsys.readouterr()

    # --update は、基準にある指標の値だけを書き換える（許容と、足していない指標はそのまま）
    assert run(_summary(0.90, 0.005), "--update") == 0
    data = json.loads(base.read_text(encoding="utf-8"))
    assert data["sets"]["demo"]["metrics"]["stft_midi"] == {"value": 0.9, "tolerance": 0.03} and "stft_self" not in data["sets"]["demo"]["metrics"]


def test_run_eval_summary_collects_per_class_values():
    rows = [
        {"id": "a", "stft_midi": 1.0, "stft_self": 0.1, "m0_1": 0.5, "part_dist_values": {"kick": (0.30, 0.25, 0.03)}, "midi_offsets_values": {"kick": 0.0, "hihat": 0.1}},
        {"id": "b", "stft_midi": 0.8, "stft_self": 0.2, "part_dist_values": {"kick": (0.20, 0.15, 0.01), "snare": (0.4, 0.5, 0.02)}, "midi_offsets_values": {"kick": 0.02}},
        {"id": "c"},  # パート別の切り出しができなかった曲
    ]
    s = run_eval.summary(rows)
    assert s["n_songs"] == 3 and s["songs"] == ["a", "b", "c"]
    assert abs(s["means"]["stft_midi"] - 0.9) < 1e-9 and s["counts"]["stft_midi"] == 2
    assert abs(s["means"]["part_dist_midi_kick"] - 0.25) < 1e-9 and abs(s["means"]["part_dist_c_kick"] - 0.20) < 1e-9
    assert abs(s["means"]["part_dist_midi_snare"] - 0.4) < 1e-9 and s["counts"]["part_dist_midi_snare"] == 1
    assert abs(s["means"]["midi_offsets_kick"] - 0.01) < 1e-9 and s["means"]["midi_offsets_hihat"] == 0.1
    assert s["per_song"]["c"] == {} and json.dumps(s)  # JSON に書ける
