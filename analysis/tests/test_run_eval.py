"""合成のステム集で run_eval が通ることを確かめる。"""

import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drumeval import save_hits_csv  # noqa: E402
from drumeval.synth import render  # noqa: E402
from tests.test_extract import _sparse_pattern  # noqa: E402


def test_run_eval_on_synthetic_set(tmp_path):
    import soundfile as sf

    sr = 22050
    for i, bleed in enumerate((0.0, 0.2)):
        d = tmp_path / "stems" / f"s{i + 1:02d}"
        d.mkdir(parents=True)
        hits = _sparse_pattern(4)
        sf.write(d / "drums.wav", render(hits, sr=sr, bleed=bleed).astype(np.float32), sr)
        if i == 0:
            save_hits_csv(hits, d / "hits.csv")
    out = tmp_path / "result.md"
    script = Path(__file__).resolve().parents[1] / "run_eval.py"
    r = subprocess.run([sys.executable, str(script), str(tmp_path / "stems"), "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    md = out.read_text(encoding="utf-8")
    assert "| s01 |" in md and "| s02 |" in md
    assert "正解付きの曲: 1 / 2" in md
    assert "M0-1" in md


def test_pseudo_truth_from_part_stems(tmp_path):
    """パート別ステム（キック.wav など）だけがあるとき、疑似正解を作って F 値を出す。"""
    import soundfile as sf

    from drumeval import load_hits, transcription_scores
    from drumeval.labels import to_coarse
    from tests.test_pipeline import _rock_pattern

    sr = 22050
    ref = _rock_pattern(6)
    dur = max(h.time for h in ref) + 1.2
    song = tmp_path / "stems" / "a01"
    song.mkdir(parents=True)
    mix = np.zeros(int(dur * sr))
    for cls, name in (("kick", "キック"), ("snare", "スネア"), ("hihat", "ハイハット")):
        part = render([h for h in ref if to_coarse(h.cls) == cls], sr=sr, duration=dur)
        sf.write(song / f"{name}.wav", part.astype(np.float32), sr)
        mix += part
    sf.write(song / "drum.wav", (mix / np.max(np.abs(mix)) * 0.9).astype(np.float32), sr)
    sf.write(song / "ギター.wav", np.zeros(sr, dtype=np.float32), sr)  # 他のステムが並んでいても drum.wav を選ぶ

    script = Path(__file__).resolve().parents[1] / "run_eval.py"
    r = subprocess.run([sys.executable, str(script), str(tmp_path / "stems")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "疑似（パート別ステム）" in r.stdout
    pseudo = load_hits(song / "hits_pseudo.csv")
    s = transcription_scores(ref, pseudo)
    for c in ("kick", "snare", "hihat"):
        assert s[c].f_measure >= 0.95, (c, s[c])
