"""E-GMD の正解の時刻のずれ: 一覧を作り（egmd_offsets.py）、学習と採点の打点に反映する。"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import egmd_offsets  # noqa: E402
from drumeval import Hit, save_hits_csv  # noqa: E402
from drumeval.egmd import OFFSETS_CSV, read_index, track_hits  # noqa: E402
from drumeval.labels import load_hits_midi, save_hits_midi  # noqa: E402
from drumeval.synth import render  # noqa: E402
from tests.test_pipeline import _rock_pattern  # noqa: E402

SR = 22050
DELAYS_MS = {"a": 0.0, "b": 40.0, "c": -30.0}


def _make_dataset(root: Path) -> None:
    """E-GMD と同じ配置の小さなデータ: 音が正解の MIDI より 0／+40／−30 ms ずれた 3 本。"""
    import soundfile as sf

    hits = [Hit(h.time + 0.3, h.cls, h.velocity) for h in _rock_pattern(6)]
    (root / "drummer1" / "session1").mkdir(parents=True)
    rows = []
    for name, delay in DELAYS_MS.items():
        x = render([Hit(h.time + delay / 1000.0, h.cls, h.velocity) for h in hits], sr=SR)
        sf.write(root / "drummer1" / "session1" / f"{name}.wav", x.astype(np.float32), SR)
        save_hits_midi(hits, root / "drummer1" / "session1" / f"{name}.midi")
        rows.append({"audio_filename": f"drummer1/session1/{name}.wav", "midi_filename": f"drummer1/session1/{name}.midi",
                     "split": "train" if name != "c" else "test", "duration": f"{x.size / SR:.2f}", "kit_name": "Kit", "style": "rock"})
    with open(root / "e-gmd-v1.0.0.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_offsets_csv_is_written_and_applied_to_labels(tmp_path, capsys):
    root = tmp_path / "egmd"
    _make_dataset(root)
    assert all(t.offset == 0.0 for t in read_index(root))  # 一覧が無ければ直さない
    assert egmd_offsets.main([str(root), "--workers", "1", "--max-tracks", "2"]) == 0  # 途中まで
    assert egmd_offsets.main([str(root), "--workers", "1"]) == 0  # 続きから
    out = capsys.readouterr().out
    assert "測り済み 2 本、これから 1 本" in out
    with open(root / OFFSETS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3  # 同じトラックを 2 度書かない
    tracks = {Path(t.midi).stem: t for t in read_index(root)}
    assert len(tracks) == 3  # label_offsets.csv を E-GMD の目録と取り違えない
    for name, delay in DELAYS_MS.items():
        assert abs(tracks[name].offset * 1000 - delay) < 2.5, (name, tracks[name].offset)
        raw = load_hits_midi(tracks[name].midi)
        fixed = track_hits(tracks[name])
        assert len(fixed) == len(raw)
        assert abs(np.median([f.time - r.time for f, r in zip(fixed, raw)]) * 1000 - delay) < 2.5  # 打点は音に合う時刻になる
    assert all(t.offset == 0.0 for t in read_index(root, use_offsets=False))  # 比較用に、直さない読み方もできる


def test_weak_estimates_are_not_applied(tmp_path):
    root = tmp_path / "egmd"
    _make_dataset(root)
    with open(root / OFFSETS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["midi_filename", "offset_ms", "sharpness", "n_hits"])
        w.writerow(["drummer1/session1/a.midi", "80.0", "1.2", "50"])  # 山がはっきりしない → 使わない
        w.writerow(["drummer1/session1/b.midi", "40.0", "4.0", "50"])
    tracks = {Path(t.midi).stem: t for t in read_index(root)}
    assert tracks["a"].offset == 0.0 and abs(tracks["b"].offset - 0.040) < 1e-9 and tracks["c"].offset == 0.0


def test_stems_mode_writes_label_offset_and_run_eval_uses_it(tmp_path, capsys):
    """run_eval.py の形のフォルダ: label_offset.txt を書き、評価では正解の打点にその値を足す。"""
    import subprocess

    import soundfile as sf

    from drumeval.stems import read_label_offset

    hits = [Hit(h.time + 0.3, h.cls, h.velocity) for h in _rock_pattern(6)]
    song = tmp_path / "stems" / "e01"
    song.mkdir(parents=True)
    x = render([Hit(h.time - 0.060, h.cls, h.velocity) for h in hits], sr=SR)  # 音が正解より 60 ms 早い（採点の許容 50 ms を超える）
    sf.write(song / "drums.wav", x.astype(np.float32), SR)
    save_hits_csv(hits, song / "hits.csv")
    (song / "start.txt").write_text("0", encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "run_eval.py"

    def m0_1() -> float:
        r = subprocess.run([sys.executable, str(script), str(tmp_path / "stems"), "--out", str(tmp_path / "r.md")], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        line = next(l for l in (tmp_path / "r.md").read_text(encoding="utf-8").splitlines() if l.startswith("| e01 |"))
        heads = [h.strip() for h in (tmp_path / "r.md").read_text(encoding="utf-8").splitlines()[2].split("|")[1:-1]]
        return float([v.strip() for v in line.split("|")[1:-1]][heads.index("M0-1")])

    before = m0_1()
    assert read_label_offset(song) == 0.0
    assert egmd_offsets.main(["--stems", str(tmp_path / "stems")]) == 0
    capsys.readouterr()
    assert abs(read_label_offset(song) * 1000 + 60.0) < 2.5
    after = m0_1()
    assert after > before + 0.3, (before, after)  # ずれたままだと、正しい検出が誤りに数えられていた
    import re

    assert re.search(r"正解（時刻を -(59|60|61) ms 直した）", (tmp_path / "r.md").read_text(encoding="utf-8"))


def test_a_hit_at_the_very_start_survives_a_tiny_negative_offset():
    """曲の頭ちょうどの打は、ずれの測定の誤差（−0.1 ms）で正解から落ちない。区間より十分前の打は落ちる。"""
    from drumeval.stems import truth_hits_in_window

    hits = [Hit(0.0, "kick", 100), Hit(0.5, "snare", 90), Hit(0.995, "hihat", 80), Hit(1.2, "kick", 100)]
    got = truth_hits_in_window(hits, start=0.0, duration=1.0, offset=-0.0001)
    assert [(round(h.time, 4), h.cls) for h in got] == [(0.0, "kick"), (0.4999, "snare"), (0.9949, "hihat")]
    # 区間を途中から始めるとき: 頭の 3 ms 前の打は頭の打として残し（時刻 0）、20 ms 前の打は落とす
    got = truth_hits_in_window([Hit(0.480, "kick", 100), Hit(0.497, "snare", 90), Hit(0.6, "kick", 100)], start=0.5, duration=1.0)
    assert [(round(h.time, 3), h.cls) for h in got] == [(0.0, "snare"), (0.1, "kick")]
