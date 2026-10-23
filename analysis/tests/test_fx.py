"""既知のエフェクトを掛けた合成テストで、当てはめが効くことを確かめる（M0-3 の形）。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drumeval import Hit, multi_resolution_stft_distance  # noqa: E402
from drumeval.fx import ChainParams, apply_chain, estimate_initial, fit_chain  # noqa: E402
from drumeval.synth import render  # noqa: E402


def _pattern(n_bars=4, bpm=110.0):
    beat = 60.0 / bpm
    hits = []
    for bar in range(n_bars):
        t0 = 0.1 + bar * 4 * beat
        hits += [Hit(t0, "kick", 110), Hit(t0 + 2 * beat, "kick", 100)]
        hits += [Hit(t0 + beat, "snare", 115), Hit(t0 + 3 * beat, "snare", 110)]
        hits += [Hit(t0 + k * beat / 2, "hihat_closed", 70) for k in range(8)]
    return hits


def test_neutral_chain_is_identity():
    sr = 22050
    x = render(_pattern(1), sr=sr)
    y = apply_chain(x, sr, ChainParams())
    assert np.allclose(x, y, atol=1e-9)


def test_eq_direction_and_initial_estimate():
    sr = 22050
    x = render(_pattern(2), sr=sr)
    p = ChainParams(low_shelf_db=-9.0, high_shelf_db=6.0, gain_db=-3.0)
    y = apply_chain(x, sr, p)
    init = estimate_initial(x, y, sr)
    assert init.low_shelf_db < -3.0
    assert init.high_shelf_db > 2.0


def test_fit_reduces_distance_clearly():
    sr = 22050
    x = render(_pattern(3), sr=sr)
    target = ChainParams(gain_db=-4.0, low_shelf_db=-6.0, peak_hz=2500.0, peak_db=5.0,
                         high_shelf_db=-5.0, reverb_rt60=0.6, reverb_damp=0.6, reverb_mix=0.3)
    y = apply_chain(x, sr, target)
    fitted, log = fit_chain(x, y, sr, max_seconds=5.0, maxiter=3)
    assert log["distance_after"] < 0.5 * log["distance_before"], log
    assert log["distance_after"] <= log["distance_init"] + 1e-9
    d = multi_resolution_stft_distance(y, apply_chain(x, sr, fitted))
    assert d["total"] < multi_resolution_stft_distance(y, x)["total"] * 0.5
    # リバーブの長さは桁が合う程度でよい
    assert 0.25 < fitted.reverb_rt60 < 1.5
