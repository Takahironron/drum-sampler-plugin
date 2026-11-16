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


def test_feature_kit_with_gain_and_choke_is_closer_than_3x3():
    """特徴で選ぶ代表音＋音量合わせ＋チョークは、3 層 × 3 音より元に近い（合成音）。"""
    from drumeval.extract import build_kit_by_features, hit_segments_from_parts, segment_features
    from drumeval.labels import to_coarse

    sr = 22050
    rng = np.random.default_rng(5)
    ref = _sparse_pattern(12)
    # 開いたハイハットを混ぜて奏法の違いを作る
    ref = [Hit(h.time, "hihat_open" if (h.cls == "hihat_closed" and rng.random() < 0.3) else h.cls, h.velocity) for h in ref]
    dur = max(h.time for h in ref) + 1.5
    parts = {}
    mix = np.zeros(int(dur * sr))
    for cls in ("kick", "snare", "hihat"):
        part = render([h for h in ref if to_coarse(h.cls) == cls], sr=sr, duration=dur, variation=0.0)
        parts[cls] = (part, sr)
        mix += part
    segs = cut_segments(x := mix, sr, ref)  # noqa: F841  （混合からの切り出しは使わない）
    from drumeval.extract import cut_segments_from_parts

    segs_p = cut_segments_from_parts(parts, ref)
    own = hit_segments_from_parts(parts, ref)
    levels = [20 * np.log10(np.max(np.abs(a)) + 1e-9) if a is not None else None for a in own]
    feats = [segment_features(a, sr) if a is not None else None for a in own]
    kit_a = build_kit(segs_p, sr)
    kit_c = build_kit_by_features(segs_p, sr, n_per_class=8)
    assert kit_c.n_samples("hihat") <= 8 and kit_c.feat_norm["hihat"]
    ya = render_kit(ref, kit_a, duration=dur)
    yc = render_kit(ref, kit_c, duration=dur, hit_levels_db=levels, hit_features=feats, gain_match=True, choke=True)
    da = multi_resolution_stft_distance(mix, ya)["total"]
    dc = multi_resolution_stft_distance(mix, yc)["total"]
    assert dc < da, (da, dc)
    # 保存と読み込みで特徴も残る
    import tempfile

    d = Path(tempfile.mkdtemp()) / "kit"
    save_kit(kit_c, d)
    back = load_kit(d)
    assert back.feat_norm.keys() == kit_c.feat_norm.keys()
    assert back.pick_nearest("kick", feats[0]) is not None


def test_onset_lag_handles_short_samples():
    from drumeval.extract import _onset_lag

    sr = 44100
    rng = np.random.default_rng(0)
    target = rng.standard_normal(2000)
    # 短い音（40 ms 未満）でも落ちない
    for n in (5, 16, 100, 587, 1500):
        lag = _onset_lag(target, rng.standard_normal(n), sr)
        assert isinstance(lag, int)
    # ずらした音は元の位置に戻る
    pulse = np.zeros(4000)
    pulse[1000:1100] = np.hanning(100)
    shifted = np.roll(pulse, 200)  # 200 サンプル遅れ
    assert _onset_lag(pulse, shifted, sr) == -200


def test_segment_limit_is_longer_for_cymbals():
    """区間の上限は種類ごと。シンバルは 4 秒まで取り、余韻を切らない（全種類 1.5 秒だと高域が減る）。"""
    from drumeval.extract import cut_segments_from_parts, hit_segments_from_parts, max_len_for, self_render_from_parts

    sr = 8000
    n = sr * 10
    rng = np.random.default_rng(0)
    cym = np.zeros(n)
    cym[sr : sr * 6] = rng.standard_normal(sr * 5) * np.exp(-np.arange(sr * 5) / (1.5 * sr))  # 1 秒目に打って 5 秒鳴る
    kick = np.zeros(n)
    kick[sr * 2 : sr * 5] = rng.standard_normal(sr * 3) * 0.1
    parts = {"cymbal": (cym, sr), "kick": (kick, sr)}
    hits = [Hit(1.0, "crash", 100), Hit(2.0, "kick", 100)]
    assert max_len_for("cymbal") == 4.0 and max_len_for("kick") == 1.5 and max_len_for("cymbal", 1.5) == 1.5
    own = hit_segments_from_parts(parts, hits)
    assert abs(own[0].size / sr - 4.0) < 0.01 and abs(own[1].size / sr - 1.5) < 0.01
    segs = {s.cls: s for s in cut_segments_from_parts(parts, hits)}
    assert segs["cymbal"].audio.size / sr > 3.9 and segs["kick"].audio.size / sr < 1.6
    y = self_render_from_parts(parts, hits, 10.0)
    assert np.sum(y[int(3.0 * sr) : int(4.9 * sr)] ** 2) > 0.5 * np.sum(cym[int(3.0 * sr) : int(4.9 * sr)] ** 2)  # 2〜4 秒後の余韻も鳴る
    short = self_render_from_parts(parts, hits, 10.0, max_len_sec=1.5)
    assert np.sum(short[int(3.0 * sr) : int(4.9 * sr)] ** 2) < 0.2 * np.sum(cym[int(3.0 * sr) : int(4.9 * sr)] ** 2)



# ---------------------------------------------------------------------------
# キットと MIDI の約束（強さ ↔ 音量、立ち上がりの位置、MIDI だけで鳴らす）
# ---------------------------------------------------------------------------

def test_velocity_and_level_round_trip():
    """強さ 1〜127 は、その種類の打の音量の範囲に dB で線形に対応する。往復しても 0.13 dB 以内。"""
    from drumeval.extract import level_for_velocity, level_range_for, velocity_for_level

    rng = level_range_for([-20.0, -8.0, -3.0])
    assert rng == (-33.0, -3.0)  # 上限は最も大きい打、下限は 30 dB 下
    assert velocity_for_level(-3.0, rng) == 127 and velocity_for_level(-33.0, rng) == 1 and velocity_for_level(-60.0, rng) == 1
    for level in (-30.0, -20.0, -11.3, -3.0):
        assert abs(level_for_velocity(velocity_for_level(level, rng), rng) - level) <= 30.0 / 126 / 2 + 1e-9
    assert level_range_for([-5.0, -5.0, -5.0]) == (-35.0, -5.0)  # 全部同じ大きさ（打ち込み）でも、強さを下げれば小さく鳴る
    assert level_range_for([-50.0, -5.0]) == (-50.0, -5.0)  # 30 dB より小さい打があれば、そこまで広げる
    assert level_range_for([]) == (-30.0, 0.0)
    assert level_for_velocity(None, rng) == -3.0


def test_refine_onsets_moves_hits_to_the_attack():
    """採譜の時刻がずれていても（±8 ms）、そのパートの音の立ち上がりに合わせ直す。音の無い打は動かさない。"""
    from drumeval.extract import refine_onsets

    sr = 44100
    rng = np.random.default_rng(0)
    true = [0.5, 0.9, 1.3, 1.7, 2.1]
    x = np.zeros(int(3.0 * sr))
    for i, t in enumerate(true):
        n = int(0.3 * sr)
        x[int(t * sr) : int(t * sr) + n] += rng.standard_normal(n) * np.exp(-np.arange(n) / (0.05 * sr)) * (0.8 if i % 2 == 0 else 0.3)
    jitter = [0.008, -0.007, 0.004, -0.008, 0.006]
    hits = [Hit(t + j, "snare", None) for t, j in zip(true, jitter)] + [Hit(2.7, "snare", None), Hit(1.0, "kick", None)]
    out = refine_onsets({"snare": (x, sr)}, hits)
    assert len(out) == len(hits)
    for h, t in zip(out[:5], true):
        assert abs(h.time - t) < 0.0015, (h.time, t)
    assert out[5].time == 2.7  # 音が無い（立ち上がりが無い）打はそのまま
    assert out[6].time == 1.0  # パートの無い種類はそのまま


def test_render_midi_picks_nearest_level_and_matches_gain(tmp_path):
    """MIDI だけで鳴らす: 強さから目標の音量を決め、音量の最も近い音を選び、音量を目標に合わせる。立ち上がりは打点に重なる。"""
    from drumeval.extract import SAMPLE_LEAD_SEC, Kit, Layer, level_range_for, load_kit, render_midi, save_kit, velocity_for_level

    sr = 22050
    lead = int(round(SAMPLE_LEAD_SEC * sr))
    n = int(0.2 * sr)
    t = np.arange(n) / sr

    def tone(freq: float, peak_db: float) -> np.ndarray:
        s = np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.05)
        s = s / np.max(np.abs(s)) * 10 ** (peak_db / 20)
        return np.concatenate([np.zeros(lead), s])  # 切り出した音と同じく、立ち上がりは頭から lead の位置

    layer = Layer(1, 127, samples=[tone(300.0, -20.0), tone(900.0, -10.0), tone(2700.0, -3.0)], source_times=[0.0, 1.0, 2.0],
                  peaks_db=[-20.0, -10.0, -3.0])
    kit = Kit(sr, {"snare": [layer]}, level_range={"snare": level_range_for([-25.0, -12.0, -3.0])})
    targets = [(-12.0, 900.0), (-3.0, 2700.0), (-25.0, 300.0), (-12.0, 900.0)]  # (音量, 選ばれるはずの音の周波数)
    hits = [Hit(0.5 + 0.5 * i, "snare", velocity_for_level(level, kit.level_range["snare"])) for i, (level, _) in enumerate(targets)]
    y = render_midi(hits, kit, duration=3.0, choke=False)
    for h, (level, freq) in zip(hits, targets):
        a = int(round(h.time * sr))
        seg = y[a : a + n]
        assert abs(20 * np.log10(np.max(np.abs(seg))) - level) < 0.3  # 音量は目標どおり（強さの刻みは 0.24 dB）
        spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
        assert abs(np.fft.rfftfreq(seg.size, 1 / sr)[int(np.argmax(spec))] - freq) < 30.0  # 音量の最も近い音が選ばれる
        assert np.max(np.abs(y[max(a - lead - 5, 0) : a - 1])) < 1e-9 and np.max(np.abs(y[a : a + 40])) > 1e-3  # 立ち上がりは打点から
    # 同じ強さが続くと同じ音（neighbours=1）。2 にすると次に近い音と交互になる
    y2 = render_midi(hits, kit, duration=3.0, choke=False, neighbours=2)
    assert not np.allclose(y, y2)
    # 約束はキットと一緒に保存される
    save_kit(kit, tmp_path / "kit")
    again = load_kit(tmp_path / "kit")
    assert again.level_range == kit.level_range
    assert np.allclose(render_midi(hits, again, duration=3.0, choke=False), y, atol=1e-5)  # WAV は 24 bit


def test_silent_hits_marks_hits_far_below_the_loud_ones():
    """その種類の大きい打より 40 dB 以上小さい打（と、区間の取れない打）だけを「音の無い打」にする。種類ごとに判断する。"""
    from drumeval.extract import silent_hits

    hits = [Hit(0.1 * i, "kick", None) for i in range(6)] + [Hit(1.0, "snare", None), Hit(1.2, "snare", None), Hit(2.0, "tom", None)]
    levels = [-3.0, -6.0, -20.0, -42.0, -50.0, None, -30.0, -31.0, None]
    #          大     大     小      ゴースト 無音   区間なし  スネアは全部小さいが、スネアの中では大きい  タムは音量が 1 つも無い
    assert silent_hits(hits, levels) == [False, False, False, False, True, True, False, False, True]
    assert silent_hits(hits, levels, floor_db=-30.0)[3] is True  # 下限を上げると、小さい打も除く
    assert silent_hits([], []) == []


def test_low_confidence_segments_do_not_become_kit_samples(tmp_path):
    """分離の確からしさの低い打（他の音と重なった打）は、代表音にしない。確からしさはキットと一緒に保存される。"""
    from drumeval.extract import Segment, attach_confidence, build_kit_by_features, confident_segments, load_kit, save_kit

    sr = 22050
    rng = np.random.default_rng(0)
    t = np.arange(int(0.2 * sr)) / sr

    def seg(i: int, conf: float | None) -> Segment:
        a = rng.standard_normal(t.size) * np.exp(-t / (0.02 + 0.002 * i)) * (0.2 + 0.03 * i)
        return Segment("hihat", 0.25 * i, a, 20 * np.log10(np.max(np.abs(a))), conf)

    confs = [0.95, 0.4, 0.9, 0.3, 0.85, 0.5, 0.97, 0.2, 0.92, 0.88]
    segs = [seg(i, c) for i, c in enumerate(confs)]
    kept = confident_segments(segs)
    assert [s.confidence for s in kept] == [0.95, 0.9, 0.85, 0.97, 0.92, 0.88]  # 0.8 以上
    kit = build_kit_by_features(segs, sr, n_per_class=24)
    assert sorted(kit.parts["hihat"][0].confidences) == sorted(c for c in confs if c >= 0.8)
    assert sorted(kit.parts["hihat"][0].source_times) == sorted(0.25 * i for i, c in enumerate(confs) if c >= 0.8)
    assert kit.n_samples("hihat") == 6 and build_kit_by_features(segs, sr, n_per_class=24, min_confidence=0.0).n_samples("hihat") == 10
    # どの打も重なっている種類では、確からしい方の半分を残す（下限は、0.8 とその種類の中央値の小さい方）
    low = [seg(i, c) for i, c in enumerate([0.3, 0.5, 0.2, 0.6, 0.4, 0.1])]
    assert sorted(s.confidence for s in confident_segments(low)) == [0.4, 0.5, 0.6]
    # 確からしさの分からない区間（分離ツールのパート別ステムから切ったときなど）は、そのまま全部が候補
    unknown = [seg(i, None) for i in range(5)]
    assert confident_segments(unknown) == unknown and confident_segments([]) == []
    # 打の並びから区間に付ける
    hits = [Hit(s.time, "hihat_closed", None) for s in unknown]
    attach_confidence(unknown, hits, [0.1, 0.2, 0.3, 0.4, 0.5])
    assert [s.confidence for s in unknown] == [0.1, 0.2, 0.3, 0.4, 0.5]
    save_kit(kit, tmp_path / "kit")
    assert load_kit(tmp_path / "kit").parts["hihat"][0].confidences == kit.parts["hihat"][0].confidences
