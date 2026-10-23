"""合成音で、切り出し → 層 → 再生 の通しを確かめる。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drumeval import Hit, multi_resolution_stft_distance  # noqa: E402
from drumeval.extract import build_kit, cut_segments, estimate_velocities, load_kit, render_kit, save_kit, select_isolated_hits  # noqa: E402
from drumeval.synth import render  # noqa: E402


def _sparse_pattern(n_bars: int = 8, bpm: float = 100.0) -> list[Hit]:
    """切り出せる打が十分にある、間の空いたパターン（強さも散らす）。"""
    beat = 60.0 / bpm
    rng = np.random.default_rng(0)
    hits = []
    for bar in range(n_bars):
        t0 = 0.1 + bar * 4 * beat
        hits.append(Hit(t0, "kick", int(rng.integers(60, 128))))
        hits.append(Hit(t0 + beat, "snare", int(rng.integers(60, 128))))
        hits.append(Hit(t0 + 2 * beat, "kick", int(rng.integers(60, 128))))
        hits.append(Hit(t0 + 3 * beat, "hihat_closed", int(rng.integers(40, 128))))
        hits.append(Hit(t0 + 3.5 * beat, "hihat_closed", int(rng.integers(40, 128))))
    return hits


def test_isolated_selection_skips_overlaps():
    hits = [Hit(0.0, "kick"), Hit(0.01, "hihat"), Hit(1.0, "snare"), Hit(1.05, "kick"), Hit(2.0, "kick")]
    iso = select_isolated_hits(hits, pre_sec=0.1, post_sec=0.15)
    assert [h.time for h, _ in iso] == [2.0]


def test_extract_and_rerender_is_close_to_original():
    sr = 44100
    ref = _sparse_pattern()
    x = render(ref, sr=sr)
    segs = cut_segments(x, sr, ref)
    assert len(segs) == len(ref)  # すべて重ならない
    kit = build_kit(segs, sr, n_layers=3, round_robin=2)
    assert set(kit.parts) == {"kick", "snare", "hihat"}
    assert len(kit.parts["kick"]) == 3
    y = render_kit(ref, kit, duration=x.size / sr)
    d_same = multi_resolution_stft_distance(x, y)
    # 種類を入れ替えて鳴らしたものより明確に近い
    swapped = [Hit(h.time, {"kick": "snare", "snare": "kick"}.get(h.cls, h.cls), h.velocity) for h in ref]
    d_swapped = multi_resolution_stft_distance(x, render_kit(swapped, kit, duration=x.size / sr))
    assert d_same["total"] < 0.5 * d_swapped["total"], (d_same, d_swapped)


def test_few_hits_reduce_layers():
    sr = 22050
    ref = [Hit(0.1, "kick", 100), Hit(1.0, "kick", 60), Hit(2.0, "snare", 110)]
    x = render(ref, sr=sr)
    kit = build_kit(cut_segments(x, sr, ref), sr, n_layers=3)
    assert len(kit.parts["kick"]) == 2
    assert len(kit.parts["snare"]) == 1


def test_velocity_estimate_tracks_level():
    sr = 22050
    ref = [Hit(0.1, "kick", 40), Hit(1.0, "kick", 90), Hit(2.0, "kick", 127)]
    x = render(ref, sr=sr)
    est = estimate_velocities([Hit(h.time, h.cls, None) for h in ref], None, x, sr)
    vs = [h.velocity for h in est]
    assert vs == sorted(vs) and vs[0] == 30 and vs[-1] == 127


def test_kit_roundtrip(tmp_path):
    sr = 22050
    ref = _sparse_pattern(2)
    x = render(ref, sr=sr)
    kit = build_kit(cut_segments(x, sr, ref), sr)
    save_kit(kit, tmp_path / "kit")
    back = load_kit(tmp_path / "kit")
    assert back.sr == sr
    assert {c: back.n_samples(c) for c in back.parts} == {c: kit.n_samples(c) for c in kit.parts}
    assert (tmp_path / "kit" / "kit.json").exists()
