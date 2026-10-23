import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drumeval import Hit  # noqa: E402
from drumeval.features import FeatureConfig, frame_targets, log_mel, mel_filterbank, pick_hits  # noqa: E402
from drumeval.synth import render  # noqa: E402


def test_mel_filterbank_shape_and_coverage():
    cfg = FeatureConfig()
    fb = mel_filterbank(cfg)
    assert fb.shape == (cfg.n_mels, cfg.n_fft // 2 + 1)
    assert (fb.sum(axis=1) > 0).all()


def test_log_mel_frames_and_targets_align():
    cfg = FeatureConfig()
    hits = [Hit(0.5, "kick", 100), Hit(1.0, "snare", 100), Hit(1.5, "hihat_closed", 80)]
    x = render(hits, sr=cfg.sr, duration=2.0)
    feat = log_mel(x, cfg)
    assert feat.shape[1] == cfg.n_mels
    assert abs(feat.shape[0] - 2.0 * cfg.frame_rate) < 3
    y = frame_targets(hits, feat.shape[0], cfg, ("kick", "snare", "hihat"))
    assert y.sum() == 6  # 3打 × 2フレーム
    # 打点のフレームで特徴量が立ち上がる
    fr = int(round(0.5 * cfg.frame_rate))
    assert feat[fr + 1].sum() > feat[fr - 3].sum()


def test_pick_hits_roundtrip():
    cfg = FeatureConfig()
    classes = ("kick", "snare", "hihat")
    hits = [Hit(0.5, "kick"), Hit(1.0, "snare"), Hit(1.0, "hihat"), Hit(1.02, "hihat")]  # 20 ms 間隔は1つにまとまる
    y = frame_targets(hits, 300, cfg, classes)
    est = pick_hits(y, cfg, classes, threshold=0.5)
    assert sorted((round(h.time, 2), h.cls) for h in est) == [(0.5, "kick"), (1.0, "hihat"), (1.0, "snare")]


def test_synth_variation_changes_timbre():
    hits = [Hit(0.1, "kick", 100)]
    a = render(hits, sr=22050, seed=1, variation=0.0)
    b = render(hits, sr=22050, seed=1, variation=1.0)
    c = render(hits, sr=22050, seed=1, variation=0.0)
    assert np.allclose(a, c)
    assert not np.allclose(a, b)
