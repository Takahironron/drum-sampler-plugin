"""NMFD のパート分けを合成音で確かめる。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drumeval import Hit  # noqa: E402
from drumeval.labels import to_coarse  # noqa: E402
from drumeval.nmfd import LEGACY, NMFDConfig, separate, snr_db  # noqa: E402
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


# ---------------------------------------------------------------------------
# 2026-11-15 の修正（聞き比べで「全体的にこもる」）を守るテスト
# ---------------------------------------------------------------------------

def _burst(n: int, sr: int, decay_sec: float, lo_hz: float, hi_hz: float, seed: int) -> np.ndarray:
    """帯域を限った雑音の一打（指数で減衰）。シンバルやハイハットの代わり。"""
    from scipy.signal import butter, lfilter

    rng = np.random.default_rng(seed)
    b, a = butter(2, [lo_hz / (sr / 2), min(hi_hz / (sr / 2), 0.99)], btype="band")
    t = np.arange(n) / sr
    return lfilter(b, a, rng.standard_normal(n)) * np.exp(-t / decay_sec)


def _thump(n: int, sr: int, f0: float = 60.0, decay_sec: float = 0.12) -> np.ndarray:
    t = np.arange(n) / sr
    return np.sin(2 * np.pi * f0 * t * (1.0 + 0.5 * np.exp(-t / 0.02))) * np.exp(-t / decay_sec)


def _place(n_total: int, sr: int, times: list[float], make) -> np.ndarray:
    y = np.zeros(n_total)
    for i, t in enumerate(times):
        a = int(t * sr)
        s = make(i)
        e = min(a + s.size, n_total)
        y[a:e] += s[: e - a]
    return y


def test_approx_matches_naive_convolution():
    """行列積にまとめた近似が、素朴な畳み込みの和と一致する。"""
    from drumeval.nmfd import _approx

    rng = np.random.default_rng(0)
    K, F, L, T = 3, 17, 4, 40
    W, H = rng.random((K, F, L)), rng.random((K, T))
    naive = np.zeros((F, T))
    for k in range(K):
        for tau in range(L):
            naive[:, tau:] += np.outer(W[k, :, tau], H[k, : T - tau])
    assert np.allclose(_approx(W, H), naive)


def test_onset_windows_reach_the_next_hit_of_the_same_class():
    from drumeval.nmfd import onset_windows

    hits = [Hit(0.0, "kick"), Hit(0.5, "kick"), Hit(2.5, "kick"), Hit(0.25, "hihat_closed")]
    cfg = NMFDConfig(window="to_next", onset_halfwidth=2, max_window_sec=(("kick", 1.0), ("hihat", 1.0)))
    wins = onset_windows(hits, frame_rate=100.0, T=400, cfg=cfg, parts=("kick", "hihat"))
    assert wins[0] == [(0, 0, 48), (48, 50, 151), (248, 250, 351)]  # 次のキックの手前まで。離れていれば上限（1 秒）まで
    assert wins[1] == [(23, 25, 126)]
    fixed = onset_windows(hits, 100.0, 400, NMFDConfig(window="fixed", post_onset_sec=(("kick", 0.3), ("hihat", 0.12))), ("kick", "hihat"))
    assert fixed[0][0] == (0, 0, 31) and fixed[1][0] == (23, 25, 38)


def test_parts_for_hits_adds_tom_only_when_enough_tom_hits():
    from drumeval.nmfd import ALL_PARTS, PARTS, parts_for_hits

    base = [Hit(0.1 * i, "kick") for i in range(8)]
    assert parts_for_hits(base + [Hit(1.0, "tom_low")] * 3) == PARTS
    assert parts_for_hits(base + [Hit(1.0 + 0.2 * i, "tom_low") for i in range(4)]) == ALL_PARTS


def test_sustain_stays_in_the_part():
    """余韻の長い音（シンバル）が、打った後もそのパートに残る。

    従来は「打点からの距離で下がる係数」を更新のたびに活性へ掛けていたので、余韻が残差に流れて再現がこもった。
    """
    from drumeval.metrics import band_balance_db

    sr = 22050
    n = int(8.0 * sr)
    cym_t = [0.5, 2.5, 4.5, 6.5]
    kick_t = [0.5 + 0.5 * i for i in range(14)]
    cym = _place(n, sr, cym_t, lambda i: 0.3 * _burst(int(1.8 * sr), sr, 0.6, 3000.0, 9000.0, i))
    kick = _place(n, sr, kick_t, lambda i: 0.8 * _thump(int(0.4 * sr), sr))
    mix = cym + kick
    hits = [Hit(t, "crash") for t in cym_t] + [Hit(t, "kick") for t in kick_t]
    out = separate(mix, sr, hits, NMFDConfig())
    kept = 10 * np.log10(np.sum(out["cymbal"] ** 2) / np.sum(cym**2))
    assert kept > -1.0, kept  # シンバルのエネルギーの 8 割以上がシンバルのパートに残る
    assert snr_db(cym, out["cymbal"]) > 10.0 and snr_db(kick, out["kick"]) > 15.0
    high = band_balance_db(mix, out["kick"] + out["cymbal"], sr, match_rms=False)["高"]
    assert high > -1.0, high  # パートの和の高域が元から 1 dB 以上減らない
    legacy = separate(mix, sr, hits, NMFDConfig(**LEGACY))
    assert np.sum(out["cymbal"] ** 2) > 1.2 * np.sum(legacy["cymbal"] ** 2)  # 従来より余韻が残る


def test_missed_kicks_do_not_leak_into_the_hihat_part():
    """キックを取りこぼしても、その低域がハイハットのパートに入らない（ひな形に低域を持たせない）。残差が受ける。"""
    from drumeval.metrics import band_powers

    sr = 22050
    n = int(6.0 * sr)
    hat_t = [0.25 * i for i in range(1, 23)]
    kick_t = [0.5 * i + 0.25 for i in range(11)]
    hat = _place(n, sr, hat_t, lambda i: 0.2 * _burst(int(0.2 * sr), sr, 0.04, 5000.0, 10000.0, i))
    kick = _place(n, sr, kick_t, lambda i: 0.8 * _thump(int(0.4 * sr), sr))
    hits = [Hit(t, "hihat_closed") for t in hat_t]  # キックは 1 つも検出されなかった
    out = separate(hat + kick, sr, hits, NMFDConfig())
    p = band_powers(out["hihat"], sr)
    assert p["低"] / sum(p.values()) < 0.05, p  # ハイハットのパートの低域は 5% 未満
    assert np.sum(out["residual"] ** 2) > 0.5 * np.sum(kick**2)  # 取りこぼしたキックは残差へ
    assert snr_db(hat, out["hihat"]) > 6.0
