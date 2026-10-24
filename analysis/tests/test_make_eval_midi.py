import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drumeval.labels import load_hits_midi, to_coarse  # noqa: E402
from make_eval_midi import STYLES, build_song, main  # noqa: E402


def test_each_style_builds_and_roundtrips():
    for style in STYLES:
        mid, sections = build_song(style, np.random.default_rng(1))
        assert sections[0][2] == "Intro" and sections[-1][2] == "Outro"
        total_bars = sum(n for _, n in style.sections)
        assert abs(sections[-1][1] - total_bars * 4 * 60.0 / style.bpm) < 1e-6
        hits = []
        for msg_t, msg in _iter(mid):
            if msg.type == "note_on" and msg.channel == 9 and msg.velocity > 0:
                hits.append(msg_t)
        assert len(hits) > total_bars * 8
        assert min(hits) >= 0 and max(hits) <= sections[-1][1] + 0.05


def _iter(mid):
    t = 0.0
    for msg in mid:
        t += msg.time
        yield t, msg


def test_main_writes_files(tmp_path):
    assert main(["--out", str(tmp_path)]) == 0
    for style in STYLES:
        d = tmp_path / style.id
        assert (d / "song.mid").exists() and (d / "hits.csv").exists() and (d / "sections.csv").exists() and (d / "README.md").exists()
    hits = load_hits_midi(tmp_path / "b01_rock" / "song.mid")
    classes = {to_coarse(h.cls) for h in hits}
    assert {"kick", "snare", "hihat", "tom", "cymbal"} <= classes
    # hits.csv と song.mid が一致する
    import csv

    with open(tmp_path / "b01_rock" / "hits.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(hits)
