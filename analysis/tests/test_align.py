"""正解の打点と音の時刻のずれを測る（E-GMD の配布物のずれを直す道具）。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drumeval import Hit  # noqa: E402
from drumeval.align import label_offset, shift_hits  # noqa: E402
from drumeval.synth import render  # noqa: E402
from tests.test_pipeline import _rock_pattern  # noqa: E402


def _shifted_audio(hits: list[Hit], sr: int, delay_ms: float, bleed: float = 0.0) -> np.ndarray:
    """打点より delay_ms だけ遅れて（負なら早く）鳴る音。"""
    lead = 0.2  # 負のずれでも頭が切れないように、全体を 0.2 秒後ろに置いてから切る
    x = render([Hit(h.time + lead + delay_ms / 1000.0, h.cls, h.velocity) for h in hits], sr=sr, bleed=bleed, reverb_tail=0.3 if bleed else 0.0)
    return x[int(lead * sr) :]


def test_label_offset_recovers_a_constant_shift():
    hits = [Hit(h.time + 0.3, h.cls, h.velocity) for h in _rock_pattern(8)]
    for sr in (22050, 44100):
        for delay in (-73.0, -33.0, 0.0, 16.0, 62.0):
            x = _shifted_audio(hits, sr, delay)
            off, sharp = label_offset(x, sr, hits)
            assert abs(off * 1000 - delay) < 2.0, (sr, delay, off * 1000)
            assert sharp > 2.0


def test_label_offset_survives_bleed_and_reverb_and_partial_labels():
    """にじみと残響があっても、打点の一部が欠けていても（採譜の取りこぼし）、ずれは測れる。"""
    hits = [Hit(h.time + 0.3, h.cls, h.velocity) for h in _rock_pattern(8)]
    x = _shifted_audio(hits, 22050, 40.0, bleed=0.3)
    rng = np.random.default_rng(0)
    kept = [h for h in hits if rng.random() < 0.6]
    off, _ = label_offset(x, 22050, kept)
    assert abs(off * 1000 - 40.0) < 3.0


def test_label_offset_needs_a_few_hits_and_shift_hits_moves_times():
    x = np.zeros(22050)
    assert label_offset(x, 22050, [Hit(0.1, "kick", 100)]) == (0.0, 0.0)
    hits = [Hit(0.01, "kick", 100), Hit(0.5, "snare", None)]
    moved = shift_hits(hits, -0.03)
    assert [h.time for h in moved] == [0.0, 0.47] and [h.cls for h in moved] == ["kick", "snare"] and moved[0].velocity == 100
