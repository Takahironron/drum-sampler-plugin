"""E-GMD と同じ配置の小さな偽データで、読み込み → 学習 → ONNX → 評価 の通しを確かめる。torch が無ければ飛ばす。"""

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drumeval import Hit  # noqa: E402
from drumeval.egmd import EGMDChunks, augment, collect_bleed_files, estimate_offset, load_audio, read_index  # noqa: E402
from drumeval.features import FeatureConfig  # noqa: E402
from drumeval.synth import render  # noqa: E402
from make_eval_midi import STYLES, build_song  # noqa: E402


def _fake_egmd(root: Path, n: int = 4, seconds: float = 12.0) -> None:
    import mido
    import soundfile as sf

    root.mkdir(parents=True)
    rows = []
    for i in range(n):
        style = STYLES[i % len(STYLES)]
        mid, _ = build_song(style, np.random.default_rng(i))
        d = root / "drummer1" / f"session{i}"
        d.mkdir(parents=True, exist_ok=True)
        midi_path = d / f"{i}.mid"
        mid.save(str(midi_path))
        # MIDI → 打点 → 合成音（44.1 kHz ステレオ、先頭 seconds 秒）
        hits = []
        t = 0.0
        for msg in mido.MidiFile(str(midi_path)):
            t += msg.time
            if msg.type == "note_on" and msg.channel == 9 and msg.velocity > 0 and t < seconds:
                from drumeval.labels import GM_DRUM_MAP

                if msg.note in GM_DRUM_MAP:
                    hits.append(Hit(t, GM_DRUM_MAP[msg.note], msg.velocity))
        x = render(hits, sr=44100, duration=seconds)
        sf.write(d / f"{i}.wav", np.stack([x, x], 1).astype(np.float32), 44100, subtype="PCM_24")
        rows.append({"drummer": "drummer1", "session": f"session{i}", "id": str(i), "style": style.name, "bpm": style.bpm,
                     "beat_type": "beat", "time_signature": "4-4", "midi_filename": f"drummer1/session{i}/{i}.mid",
                     "audio_filename": f"drummer1/session{i}/{i}.wav", "duration": seconds,
                     "split": "train" if i < n - 1 else "validation", "kit_name": f"kit{i % 2}"})
    with open(root / "e-gmd-v1.0.0.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_index_audio_and_chunks(tmp_path):
    root = tmp_path / "e-gmd"
    _fake_egmd(root)
    tracks = read_index(root)
    assert len(tracks) == 4 and sum(t.split == "train" for t in tracks) == 3
    cfg = FeatureConfig()
    x = load_audio(tracks[0].audio, cfg.sr, 1.0, 2.0)
    assert x.ndim == 1 and abs(x.size - 2.0 * cfg.sr) <= 2
    ds = EGMDChunks(tracks[:3], cfg, chunk_sec=4.0, chunks_per_track=2, do_augment=True)
    assert len(ds) == 6
    feat, y = ds[0]
    assert feat.shape[1] == cfg.n_mels and y.shape == (feat.shape[0], 5)
    assert y.sum() > 0
    # 音声と MIDI のずれは小さい
    hits = ds.hits_of(tracks[0])
    off = estimate_offset(load_audio(tracks[0].audio, cfg.sr, 0.0, 10.0), cfg.sr, hits, cfg)
    assert abs(off) <= 0.02


def test_augment_and_bleed(tmp_path):
    import soundfile as sf

    bleed_dir = tmp_path / "stems" / "a01"
    bleed_dir.mkdir(parents=True)
    sr = 22050
    sf.write(bleed_dir / "ベース.wav", (0.3 * np.sin(2 * np.pi * 55 * np.arange(sr * 3) / sr)).astype(np.float32), sr)
    sf.write(bleed_dir / "drum.wav", np.zeros(sr, dtype=np.float32), sr)
    files = collect_bleed_files(tmp_path / "stems")
    assert [f.name for f in files] == ["ベース.wav"]
    x = render([Hit(0.1, "kick", 100), Hit(0.6, "snare", 100)], sr=sr, duration=2.0)
    y = augment(x, sr, np.random.default_rng(3), files)
    assert y.shape == x.shape and np.max(np.abs(y)) <= 1.0 and not np.allclose(x, y)


def test_train_egmd_and_eval(tmp_path):
    root = tmp_path / "e-gmd"
    _fake_egmd(root)
    out = tmp_path / "m.onnx"
    script = Path(__file__).resolve().parents[1] / "train_transcriber.py"
    r = subprocess.run([sys.executable, str(script), "--egmd", str(root), "--epochs", "1", "--batch", "2", "--seconds", "3",
                        "--chunks-per-track", "1", "--workers", "0", "--hidden", "16", "--val-tracks", "1", "--val-seconds", "5",
                        "--max-tracks", "3", "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.exists() and out.with_suffix(".json").exists() and out.with_suffix(".pt").exists()
    assert "E-GMD: 学習 3 本" in r.stdout
    ev = Path(__file__).resolve().parents[1] / "eval_egmd.py"
    r = subprocess.run([sys.executable, str(ev), str(out), str(root), "--split", "validation", "--seconds", "5", "--check-offset"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "M0-1" in r.stdout and "ずれの見積もり" in r.stdout
