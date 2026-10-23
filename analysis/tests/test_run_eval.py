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
