"""合成データで採点スクリプトの動きを確かめる。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drumeval import Hit, load_hits, m0_transcription_score, multi_resolution_stft_distance, save_hits_csv, transcription_scores  # noqa: E402
from drumeval.labels import load_hits_midi  # noqa: E402


def _pattern(n_bars: int = 4, bpm: float = 120.0) -> list[Hit]:
    beat = 60.0 / bpm
    hits = []
    for bar in range(n_bars):
        t0 = bar * 4 * beat
        hits += [Hit(t0, "kick", 110), Hit(t0 + 2 * beat, "kick", 100)]
        hits += [Hit(t0 + beat, "snare", 115), Hit(t0 + 3 * beat, "snare", 115)]
        hits += [Hit(t0 + k * beat / 2, "hihat_closed", 70) for k in range(8)]
    return hits


def test_perfect_estimate_scores_one():
    ref = _pattern()
    assert m0_transcription_score(ref, ref) == pytest.approx(1.0)


def test_jitter_within_tolerance_still_matches():
    ref = _pattern()
    est = [Hit(h.time + 0.03, h.cls, h.velocity) for h in ref]
    assert m0_transcription_score(ref, est) == pytest.approx(1.0)
    est = [Hit(h.time + 0.08, h.cls, h.velocity) for h in ref]
    assert m0_transcription_score(ref, est) == pytest.approx(0.0)


def test_missing_class_lowers_only_that_class():
    ref = _pattern()
    est = [h for h in ref if h.cls != "snare"]
    s = transcription_scores(ref, est)
    assert s["kick"].f_measure == pytest.approx(1.0)
    assert s["hihat"].f_measure == pytest.approx(1.0)
    assert s["snare"].f_measure == pytest.approx(0.0)
    assert s["snare"].n_est == 0
    # 正解も推定も無い種類は 1
    assert s["tom"].f_measure == pytest.approx(1.0)
    assert m0_transcription_score(ref, est) == pytest.approx(2 / 3)


def test_fine_classes_fold_into_coarse():
    ref = [Hit(0.0, "hihat_closed"), Hit(0.5, "hihat_open"), Hit(1.0, "ride_bell")]
    est = [Hit(0.0, "hihat"), Hit(0.5, "hihat"), Hit(1.0, "cymbal")]
    s = transcription_scores(ref, est)
    assert s["hihat"].f_measure == pytest.approx(1.0)
    assert s["cymbal"].f_measure == pytest.approx(1.0)


def test_csv_roundtrip(tmp_path):
    ref = _pattern(1)
    p = tmp_path / "hits.csv"
    save_hits_csv(ref, p)
    back = load_hits(p)
    assert [(round(h.time, 6), h.cls, h.velocity) for h in back] == [(round(h.time, 6), h.cls, h.velocity) for h in sorted(ref, key=lambda h: h.time)]


def test_midi_gm_map(tmp_path):
    mido = pytest.importorskip("mido")
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    tr.append(mido.Message("note_on", channel=9, note=36, velocity=100, time=0))
    tr.append(mido.Message("note_off", channel=9, note=36, velocity=0, time=120))
    tr.append(mido.Message("note_on", channel=9, note=42, velocity=60, time=120))  # 0.25 秒
    tr.append(mido.Message("note_off", channel=9, note=42, velocity=0, time=120))
    tr.append(mido.Message("note_on", channel=9, note=38, velocity=120, time=120))  # 0.5 秒
    tr.append(mido.Message("note_off", channel=9, note=38, velocity=0, time=120))
    p = tmp_path / "x.mid"
    mid.save(str(p))
    hits = load_hits_midi(p)
    assert [(round(h.time, 3), h.cls, h.velocity) for h in hits] == [(0.0, "kick", 100), (0.25, "hihat_closed", 60), (0.5, "snare", 120)]


def test_stft_distance_zero_for_identical_and_grows_with_difference():
    rng = np.random.default_rng(0)
    sr = 44100
    t = np.arange(sr) / sr
    x = np.sin(2 * np.pi * 110 * t) * np.exp(-6 * t) + 0.01 * rng.standard_normal(sr)
    d0 = multi_resolution_stft_distance(x, x)
    assert d0["total"] == pytest.approx(0.0, abs=1e-9)
    y_small = x + 0.02 * rng.standard_normal(sr)
    y_big = x + 0.2 * rng.standard_normal(sr)
    d_small = multi_resolution_stft_distance(x, y_small)
    d_big = multi_resolution_stft_distance(x, y_big)
    assert 0 < d_small["total"] < d_big["total"]


def test_stft_distance_detects_eq_change():
    """ローパスを掛けると距離が増え、掛け直して戻すと減る（M0-3 の形の確認）。"""
    from scipy.signal import butter, lfilter

    rng = np.random.default_rng(1)
    sr = 44100
    x = rng.standard_normal(sr)  # 白色雑音
    b, a = butter(2, 2000 / (sr / 2))
    y = lfilter(b, a, x)
    d_filtered = multi_resolution_stft_distance(x, y)
    d_same = multi_resolution_stft_distance(x, x)
    assert d_filtered["total"] > d_same["total"] + 0.1
