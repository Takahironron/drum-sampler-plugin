"""打点ラベルから合成のドラム音を描画する。

目的は2つ。

1. 採点の通しテスト（ラベル → 音 → 採譜 → F 値）。
2. 「自社のサンプルとモーダル音源で合成データを作って学習する」道（計画書 3.2）の最小版。
   いまはキック・スネア・ハイハットの簡単な合成音だけで、`plugin/Source/BuiltInSounds.cpp` と同じ考え方。

音の質は問わない。採譜の難しさを増やすときは `bleed`（他楽器のにじみの代わりの帯域雑音）と
`reverb_tail`（単純な指数減衰の残響）を使う。
"""

from __future__ import annotations

import numpy as np

from .labels import Hit, to_coarse


def _kick(sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(0.5 * sr)
    t = np.arange(n) / sr
    freq = 45.0 + 110.0 * np.exp(-t * 35.0)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    env = np.exp(-t * 7.0)
    click = np.exp(-t * 400.0) * 0.4
    return np.tanh(1.6 * (np.sin(phase) * env + click)) * 0.9


def _snare(sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(0.35 * sr)
    t = np.arange(n) / sr
    body = (0.6 * np.sin(2 * np.pi * 185.0 * t) + 0.4 * np.sin(2 * np.pi * 330.0 * t)) * np.exp(-t * 25.0)
    snares = rng.uniform(-1, 1, n) * np.exp(-t * 14.0) * 0.6
    return np.tanh(body + snares) * 0.8


def _hihat(sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(0.12 * sr)
    t = np.arange(n) / sr
    x = rng.uniform(-1, 1, n)
    # 1次ハイパス（6 kHz）
    rc = 1.0 / (2 * np.pi * 6000.0)
    a = rc / (rc + 1.0 / sr)
    y = np.empty(n)
    prev_in = prev_out = 0.0
    for i in range(n):
        prev_out = a * (prev_out + x[i] - prev_in)
        prev_in = x[i]
        y[i] = prev_out
    return y * np.exp(-t * 45.0) * 0.5


def _tom(sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(0.4 * sr)
    t = np.arange(n) / sr
    freq = 110.0 + 40.0 * np.exp(-t * 20.0)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    return np.sin(phase) * np.exp(-t * 9.0) * 0.8


def _cymbal(sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(1.0 * sr)
    t = np.arange(n) / sr
    x = rng.uniform(-1, 1, n)
    rc = 1.0 / (2 * np.pi * 3000.0)
    a = rc / (rc + 1.0 / sr)
    y = np.empty(n)
    prev_in = prev_out = 0.0
    for i in range(n):
        prev_out = a * (prev_out + x[i] - prev_in)
        prev_in = x[i]
        y[i] = prev_out
    return y * np.exp(-t * 3.5) * 0.4


_GENERATORS = {"kick": _kick, "snare": _snare, "hihat": _hihat, "tom": _tom, "cymbal": _cymbal}


def render(
    hits: list[Hit],
    sr: int = 44100,
    duration: float | None = None,
    bleed: float = 0.0,
    reverb_tail: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """打点ラベルをモノラルの波形にする。

    - `bleed`: 0〜1。ベースの残りを模した低域の雑音と、全帯域の薄い雑音を足す。
    - `reverb_tail`: 秒。0 より大きければ指数減衰の雑音で畳み込む（残響の代わり）。
    """
    rng = np.random.default_rng(seed)
    samples = {k: g(sr, rng) for k, g in _GENERATORS.items()}
    last = max((h.time for h in hits), default=0.0)
    if duration is None:
        duration = last + 1.2
    n = int(duration * sr)
    out = np.zeros(n)
    for h in hits:
        s = samples.get(to_coarse(h.cls))
        if s is None:
            continue
        gain = 1.0 if h.velocity is None else max(h.velocity, 1) / 127.0
        start = int(round(h.time * sr))
        end = min(start + s.size, n)
        if start >= n:
            continue
        out[start:end] += s[: end - start] * gain

    if reverb_tail > 0:
        m = int(reverb_tail * sr)
        ir = rng.standard_normal(m) * np.exp(-np.arange(m) / sr * (6.9 / reverb_tail))
        ir /= np.sqrt(np.sum(ir**2)) + 1e-9
        from scipy.signal import fftconvolve

        out = out + 0.3 * fftconvolve(out, ir)[:n]

    if bleed > 0:
        from scipy.signal import butter, lfilter

        low = lfilter(*butter(2, 200 / (sr / 2)), rng.standard_normal(n)) * 2.0
        out = out + bleed * (0.5 * low + 0.05 * rng.standard_normal(n))

    peak = np.max(np.abs(out)) + 1e-9
    return out / peak * 0.9 if peak > 0.9 else out
