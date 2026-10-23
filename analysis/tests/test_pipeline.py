"""ラベル → 合成音 → 基準値の採譜 → 採点 の通しテスト。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drumeval import Hit, m0_transcription_score, transcription_scores  # noqa: E402
from drumeval.baseline import transcribe  # noqa: E402
from drumeval.synth import render  # noqa: E402


def _rock_pattern(n_bars: int = 8, bpm: float = 120.0) -> list[Hit]:
    beat = 60.0 / bpm
    hits = []
    for bar in range(n_bars):
        t0 = 0.1 + bar * 4 * beat  # 先頭に 0.1 秒の無音
        hits += [Hit(t0, "kick", 110), Hit(t0 + 2 * beat, "kick", 100), Hit(t0 + 2.5 * beat, "kick", 90)]
        hits += [Hit(t0 + beat, "snare", 115), Hit(t0 + 3 * beat, "snare", 115)]
        hits += [Hit(t0 + k * beat / 2, "hihat_closed", 70) for k in range(8)]
    return hits


def test_render_has_expected_length_and_level():
    hits = _rock_pattern(1)
    x = render(hits, sr=22050)
    assert x.ndim == 1
    assert x.size == int((max(h.time for h in hits) + 1.2) * 22050)
    assert 0.3 < np.max(np.abs(x)) <= 0.9 + 1e-9


def test_baseline_on_clean_synthetic_pattern():
    """きれいな合成音なら、基準値でも kick/snare/hihat の F 値が高いはず。"""
    sr = 44100
    ref = _rock_pattern()
    x = render(ref, sr=sr)
    est = transcribe(x, sr)
    scores = transcription_scores(ref, est)
    for c in ("kick", "snare", "hihat"):
        assert scores[c].f_measure >= 0.9, (c, scores[c])
    assert m0_transcription_score(ref, est) >= 0.9


def test_baseline_degrades_but_runs_with_bleed_and_reverb():
    sr = 44100
    ref = _rock_pattern()
    x = render(ref, sr=sr, bleed=0.5, reverb_tail=0.4, seed=3)
    est = transcribe(x, sr)
    # 走ること、キックは拾えることだけ確かめる（精度の主張はしない）
    assert len(est) > 0
    assert transcription_scores(ref, est)["kick"].recall >= 0.7
