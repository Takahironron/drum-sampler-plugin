"""学習の増強に使う「にじみ」を自作の合成音で作る（権利の明確な音だけ）。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import make_bleed_set  # noqa: E402
from drumeval.bleed import consonant, keys, midi_hz, pluck, render_bleed, vowel  # noqa: E402
from drumeval.egmd import augment, collect_bleed_files  # noqa: E402
from drumeval.metrics import band_powers  # noqa: E402

SR = 22050


def test_render_bleed_is_deterministic_and_never_silent():
    a = render_bleed(8.0, SR, np.random.default_rng(3))
    b = render_bleed(8.0, SR, np.random.default_rng(3))
    c = render_bleed(8.0, SR, np.random.default_rng(4))
    assert a.dtype == np.float32 and a.size == 8 * SR and np.array_equal(a, b) and not np.array_equal(a, c)
    for seed in range(12):  # どの乱数でも、何かの楽器が鳴っている
        y = render_bleed(6.0, SR, np.random.default_rng(seed))
        assert np.all(np.isfinite(y)) and 0.5 < np.max(np.abs(y)) <= 0.9001 and np.sqrt(np.mean(y**2)) > 0.02


def test_instruments_have_the_expected_pitch_and_band():
    rng = np.random.default_rng(0)
    for fn, note in ((pluck, 52), (keys, 60)):
        y = fn(midi_hz(note), 1.0, SR, rng)
        spec = np.abs(np.fft.rfft(y * np.hanning(y.size)))
        f = np.fft.rfftfreq(y.size, 1 / SR)
        ratio = f[int(np.argmax(spec))] / midi_hz(note)  # 最も強い成分は、基音か低い倍音（弾き方によって 2 倍音が勝つことがある）
        assert 1 <= round(ratio) <= 4 and abs(ratio - round(ratio)) < 0.04, (fn.__name__, ratio)
    c = consonant(0.1, SR, rng)
    bp = band_powers(c, SR)
    assert (bp["高"] + bp["超高"] + bp["中"]) / sum(bp.values()) > 0.95 and bp["低"] / sum(bp.values()) < 0.01  # 子音は中〜高域の雑音
    v = vowel(midi_hz(60), 0.5, SR, rng)
    assert np.all(np.isfinite(v)) and np.max(np.abs(v)) > 0


def test_cli_writes_files_that_the_trainer_picks_up(tmp_path, capsys):
    import soundfile as sf

    out = tmp_path / "bleed_synth"
    assert make_bleed_set.main([str(out), "--minutes", "1", "--seconds", "10"]) == 0
    capsys.readouterr()
    files = collect_bleed_files(out)  # 学習が「にじみ」として拾う（ドラム系の名前として除かれない）
    assert [p.name for p in files] == [f"bleed_{i:04d}.flac" for i in range(1, 7)]
    assert (out / "README.txt").exists()
    x, sr = sf.read(files[0])
    assert sr == SR and x.size == 10 * SR
    highs = []
    for p in files:
        bp = band_powers(sf.read(p)[0], SR)
        highs.append((bp["中"] + bp["高"] + bp["超高"]) / sum(bp.values()))
    assert max(highs) > 0.05  # 中〜高域にも音がある本がある
    # 増強に通る（ドラムの音に小さく混ざる）
    drums = np.random.default_rng(0).standard_normal(4 * SR).astype(np.float32) * 0.1
    y = augment(drums, SR, np.random.default_rng(1), files)
    assert y.shape == drums.shape and np.all(np.isfinite(y))
