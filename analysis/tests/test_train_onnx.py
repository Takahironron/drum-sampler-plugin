"""学習 → ONNX → ONNX Runtime の通し（短い学習。精度は見ない）。torch が無ければ飛ばす。"""

import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")


def test_train_export_and_infer(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "m.onnx"
    r = subprocess.run([sys.executable, str(root / "train_transcriber.py"), "--steps", "6", "--batch", "2",
                        "--seconds", "3", "--eval-every", "6", "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.exists() and out.with_suffix(".json").exists()
    assert "torch と ORT の最大差" in r.stdout
    diff = float(r.stdout.split("最大差")[1].split()[0])
    assert diff < 1e-4

    import numpy as np
    import soundfile as sf

    sys.path.insert(0, str(root))
    from drumeval.synth import render
    from drumeval import Hit

    wav = tmp_path / "x.wav"
    sf.write(wav, render([Hit(0.2, "kick", 100), Hit(0.7, "snare", 100)], sr=44100).astype(np.float32), 44100)
    r = subprocess.run([sys.executable, str(root / "transcribe_onnx.py"), str(out), str(wav), str(tmp_path / "e.csv")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "e.csv").exists()
