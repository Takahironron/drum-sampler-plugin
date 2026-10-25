import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drumeval.labels import load_hits_midi  # noqa: E402
from remap_midi_notes import PRESETS, main  # noqa: E402


def test_remap_keeps_timing_and_changes_notes(tmp_path):
    src = Path(__file__).resolve().parents[1] / "eval_midi" / "b01_rock" / "song.mid"
    out = tmp_path / "out.mid"
    assert main([str(src), str(out), "--preset", "maschine16"]) == 0
    import mido

    a = [(round(t, 6), m.note) for t, m in _notes(mido.MidiFile(str(src)))]
    b = [(round(t, 6), m.note) for t, m in _notes(mido.MidiFile(str(out)))]
    assert len(a) == len(b)
    assert [t for t, _ in a] == [t for t, _ in b]  # 時刻は同じ
    assert [n for _, n in a] != [n for _, n in b]  # ノートは変わる
    assert all(n == PRESETS["maschine16"].get(m, m) for (_, m), (_, n) in zip(a, b))
    # 元の正解は変わらない
    assert len(load_hits_midi(src)) == len(a)


def _notes(mid):
    t = 0.0
    for msg in mid:
        t += msg.time
        if msg.type == "note_on" and msg.channel == 9 and msg.velocity > 0:
            yield t, msg
