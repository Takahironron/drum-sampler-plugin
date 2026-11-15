"""パート別ステムの打点検出（疑似正解の作り方）のテスト。

2026-11-15 に作り直した。それまでは、余韻の長いキックの途中や他の太鼓の漏れを打点として数え、本当のキックの単独録音で
3 倍に数え過ぎていた。数え過ぎた打点は、疑似正解に対する F 値と、疑似ラベルでの微調整を歪める。
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drumeval.baseline import onsets_from_part  # noqa: E402

SR = 44100


def _kick(n: int, level: float = 1.0, seed: int = 0) -> np.ndarray:
    """余韻の長いキック: 下がっていく低い正弦波（0.25 秒で 1/e）に、近い周波数の部分音（うなり）を重ねる。"""
    t = np.arange(n) / SR
    f = 50.0 + 40.0 * np.exp(-t / 0.03)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.5 * np.sin(2 * np.pi * 61.0 * t + 0.3 * seed)
    click = 0.3 * np.random.default_rng(seed).standard_normal(n) * np.exp(-t / 0.003)
    return level * (body * np.exp(-t / 0.25) + click)


def _noise_burst(n: int, level: float, seed: int, decay: float = 0.04) -> np.ndarray:
    t = np.arange(n) / SR
    return level * np.random.default_rng(seed).standard_normal(n) * np.exp(-t / decay)


def _place(total: int, times: list[float], make) -> np.ndarray:
    y = np.zeros(total)
    for i, t in enumerate(times):
        a = int(t * SR)
        s = make(i)
        e = min(a + s.size, total)
        y[a:e] += s[: e - a]
    return y


def _matched(found: np.ndarray, truth: list[float], tol: float = 0.015) -> int:
    return sum(1 for t in truth if found.size and np.min(np.abs(found - t)) <= tol)


def _beating_kick(n: int, level: float, seed: int) -> np.ndarray:
    """余韻がうなるキック: 近い周波数の 2 つの部分音（55 Hz と 69 Hz、大きさ 1 対 0.6）が 0.3 秒で 1/e に減る。
    うなりの谷から山へ戻るところは、短い時間で見ると「立ち上がり」に見える。"""
    t = np.arange(n) / SR
    body = np.sin(2 * np.pi * 55.0 * t) + 0.6 * np.sin(2 * np.pi * 69.0 * t + 1.0 + seed)
    click = 0.3 * np.random.default_rng(seed).standard_normal(n) * np.exp(-t / 0.003)
    return level * (body / 1.6 * np.exp(-t / 0.3) + click)


def test_long_decay_kick_is_counted_once_per_hit():
    """余韻が長く、うなりのあるキックでも、1 打を 1 回だけ数える（余韻の途中を打点にしない）。雑音の床があってもよい。

    作り直す前の検出（約 12 ms 前より 6 dB 大きければ打点）は、この 8 打を 53 個と数えた。
    """
    times = [0.5 + 0.9 * i for i in range(8)]
    n = int(8.2 * SR)
    x = _place(n, times, lambda i: _beating_kick(int(1.5 * SR), 0.8, i)) + 1e-3 * np.random.default_rng(9).standard_normal(n)
    found = onsets_from_part(x, SR, cls="kick")
    assert found.size == 8, found
    assert _matched(found, times) == 8
    assert onsets_from_part(x, SR).size == 8  # 種類を渡さなくても（帯域を限らなくても）数え過ぎない


def test_leak_from_other_drums_is_not_counted_but_soft_hits_are():
    """キックのステムに他の太鼓の音が小さく漏れていても数えない。弱いキック（−20 dB）は数える。"""
    kicks = [0.5 + 1.0 * i for i in range(6)]
    ghosts = [1.0 + 2.0 * i for i in range(3)]  # −20 dB の弱いキック
    leaks = [0.75 + 0.5 * i for i in range(11)]  # スネアやハイハットの漏れ（広い帯域の短い雑音、キックより 35 dB 小さい）
    n = int(7.0 * SR)
    x = (_place(n, kicks, lambda i: _kick(int(0.4 * SR), 0.8, i))
         + _place(n, ghosts, lambda i: _kick(int(0.4 * SR), 0.08, 10 + i))
         + _place(n, leaks, lambda i: _noise_burst(int(0.15 * SR), 0.8 * 10 ** (-35 / 20), 20 + i)))
    found, levels = onsets_from_part(x, SR, cls="kick", return_levels=True)
    assert _matched(found, kicks) == 6 and _matched(found, ghosts) == 3
    assert found.size == 9, found  # 漏れは 1 つも数えない
    soft = [lv for t, lv in zip(found, levels) if any(abs(t - g) <= 0.015 for g in ghosts)]
    assert all(-26.0 < lv < -14.0 for lv in soft), soft  # 大きさは大きい打からの相対値（約 −20 dB）
    assert max(levels) < 3.0


def test_hit_after_silence_is_not_detected_early():
    """完全な無音の後の打を、早く拾わない（打点より前を見る包絡だと、裾を立ち上がりと取り違えて 80 ms 早くなる）。"""
    times = [1.0, 2.5, 4.0]
    x = _place(int(5.0 * SR), times, lambda i: _kick(int(0.5 * SR), 0.8, i))
    for cls in ("kick", None):
        found = onsets_from_part(x, SR, cls=cls)
        assert found.size == 3 and np.max(np.abs(found - np.array(times))) < 0.01, (cls, found)


def test_dense_hihats_with_accents():
    """16 分のハイハット（強弱が交互、12 dB 差）を全部拾う。"""
    times = [0.2 + 0.125 * i for i in range(32)]
    x = _place(int(4.5 * SR), times, lambda i: _noise_burst(int(0.1 * SR), 0.5 if i % 2 == 0 else 0.125, i, decay=0.015))
    found = onsets_from_part(x, SR, cls="hihat")
    assert found.size == 32 and _matched(found, times) == 32


def test_empty_and_silent_inputs():
    assert onsets_from_part(np.zeros(SR), SR, cls="kick").size == 0
    assert onsets_from_part(np.zeros(10), SR).size == 0
    t, lv = onsets_from_part(np.zeros(SR), SR, return_levels=True)
    assert t.size == 0 and lv.size == 0
