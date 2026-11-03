"""NMFD のパート分けを合成音で確かめる。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drumeval import Hit  # noqa: E402
from drumeval.labels import to_coarse  # noqa: E402
from drumeval.nmfd import NMFDConfig, separate, snr_db  # noqa: E402
from drumeval.synth import render  # noqa: E402
from tests.test_extract import _sparse_pattern  # noqa: E402


def test_separate_recovers_parts_and_sums_to_mix():
    sr = 22050
    ref = _sparse_pattern(8)
    dur = max(h.time for h in ref) + 1.2
    parts = {}
    mix = np.zeros(int(dur * sr))
    for cls in ("kick", "snare", "hihat"):
        p = render([h for h in ref if to_coarse(h.cls) == cls], sr=sr, duration=dur, variation=0.0)
        parts[cls] = p
        mix += p
    out = separate(mix, sr, ref, NMFDConfig(iterations=20, residual=False), parts=("kick", "snare", "hihat"))
    assert np.max(np.abs(sum(out.values()) - mix)) < 1e-3  # パートの和は元に等しい
    for cls in parts:
        assert snr_db(parts[cls], out[cls]) > 6.0, (cls, snr_db(parts[cls], out[cls]))
        assert snr_db(parts[cls], out[cls]) > snr_db(parts[cls], mix)  # 混ざったままより良い
