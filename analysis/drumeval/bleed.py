"""採譜モデルの学習の増強に使う「にじみ」（分離で残る、ドラム以外の楽器の音）を、自作の合成音で作る。

分離ツールが作ったドラムのステムには、ベース・ギター・鍵盤・ボーカルの音が小さく残る。学習のとき、E-GMD のドラムの音に
ドラム以外の音を小さく混ぜておくと、それをドラムと取り違えにくくなる（`egmd.augment`）。v5 まではこの材料に区分 A（CD 曲）の
ステムを使ったので、モデルを製品に入れられなかった（R3）。ここで作る音は、すべてこのファイルのコードが合成する
（音声の素材も、既存の曲の MIDI も使わない）ので、権利の疑いが無い。

楽器: ベース、撥弦（Karplus–Strong。ギターのストロークとアルペジオ）、鍵盤（減衰する倍音）、パッド、リード、声（母音のフォルマントと、
「サ行・タ行」のような子音の雑音。子音はハイハットと取り違えやすいので入れる）。テンポ・調・コード進行・編成を乱数で決める。
音楽としての出来は問わない。ドラムの音（打楽器の音色）は入れない。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirpeak, lfilter

MAJOR = (0, 2, 4, 5, 7, 9, 11)
MINOR = (0, 2, 3, 5, 7, 8, 10)
PROGRESSIONS = ((0, 4, 5, 3), (0, 5, 3, 4), (1, 4, 0, 5), (0, 3, 4, 3), (5, 3, 0, 4), (0, 2, 3, 4), (0, 0, 3, 4), (5, 4, 3, 4))
VOWELS = ((730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240), (530, 1840, 2480), (570, 840, 2410))  # あ・い・う・え・お に近いフォルマント（Hz）


def midi_hz(note: float) -> float:
    return 440.0 * 2.0 ** ((note - 69.0) / 12.0)


def _env(n: int, sr: int, attack: float, release: float) -> np.ndarray:
    e = np.ones(n)
    a = min(max(int(attack * sr), 1), n)
    r = min(max(int(release * sr), 1), n)
    e[:a] *= np.linspace(0.0, 1.0, a)
    e[-r:] *= np.linspace(1.0, 0.0, r)
    return e


def bass(f0: float, dur: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    y = np.sin(2 * np.pi * f0 * t) + rng.uniform(0.2, 0.6) * np.sin(4 * np.pi * f0 * t) + rng.uniform(0.0, 0.3) * np.sin(6 * np.pi * f0 * t)
    y *= np.exp(-t / rng.uniform(0.25, 1.2))
    return np.tanh(rng.uniform(1.0, 3.0) * y) * _env(n, sr, 0.004, 0.02)


def pluck(f0: float, dur: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """撥弦（Karplus–Strong）: 雑音の短い列を、弦の長さの遅延と弱いローパスで回す。"""
    n = int(dur * sr)
    period = max(int(round(sr / f0)), 2)
    x = np.zeros(n)
    burst = rng.uniform(-1.0, 1.0, min(period, n))
    b, a = butter(1, rng.uniform(0.2, 0.9))  # 弾く位置・強さの違い（明るさ）
    x[: burst.size] = lfilter(b, a, burst)
    rho = rng.uniform(0.990, 0.998)
    den = np.zeros(period + 2)
    den[0] = 1.0
    den[period] = den[period + 1] = -0.5 * rho
    return lfilter([1.0], den, x) * _env(n, sr, 0.001, 0.03)


def keys(f0: float, dur: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """鍵盤: 少し非調和な倍音が、高いものほど速く減衰する。"""
    n = int(dur * sr)
    t = np.arange(n) / sr
    inharm = rng.uniform(0.0, 4e-4)
    y = np.zeros(n)
    for k in range(1, 9):
        f = f0 * k * np.sqrt(1.0 + inharm * k * k)
        if f >= 0.45 * sr:
            break
        y += np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) * k ** -1.2 * np.exp(-t * (1.5 + rng.uniform(0.8, 1.6) * k))
    return y * _env(n, sr, 0.002, 0.03)


def _saw(f0: float, n: int, sr: int, phase: float = 0.0) -> np.ndarray:
    return 2.0 * (((np.arange(n) * f0 / sr) + phase) % 1.0) - 1.0


def pad(f0: float, dur: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(dur * sr)
    y = sum(_saw(f0 * 2.0 ** (d / 1200.0), n, sr, rng.random()) for d in (-rng.uniform(4, 12), 0.0, rng.uniform(4, 12)))
    b, a = butter(2, min(rng.uniform(600.0, 3000.0) / (sr / 2), 0.9))
    return lfilter(b, a, y) * _env(n, sr, rng.uniform(0.05, 0.4), rng.uniform(0.1, 0.5))


def lead(f0: float, dur: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    vib = 1.0 + rng.uniform(0.0, 0.006) * np.sin(2 * np.pi * rng.uniform(4.5, 6.5) * t)
    phase = 2 * np.pi * np.cumsum(f0 * vib) / sr
    y = (1.0 - (w := rng.uniform(0.2, 0.8))) * np.sin(phase) + w * np.sign(np.sin(phase))
    b, a = butter(2, min(rng.uniform(1500.0, 8000.0) / (sr / 2), 0.9))
    return lfilter(b, a, y) * _env(n, sr, 0.01, 0.05)


def vowel(f0: float, dur: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """声の母音: ノコギリ波を 3 つのフォルマントの共鳴に通す。少し息の雑音を足す。"""
    n = int(dur * sr)
    t = np.arange(n) / sr
    vib = 1.0 + rng.uniform(0.002, 0.01) * np.sin(2 * np.pi * rng.uniform(4.5, 6.5) * t)
    src = 2.0 * ((np.cumsum(f0 * vib) / sr) % 1.0) - 1.0 + 0.05 * rng.standard_normal(n)
    formants = VOWELS[int(rng.integers(len(VOWELS)))]
    y = np.zeros(n)
    for i, f in enumerate(formants):
        f = f * rng.uniform(0.9, 1.15)
        if f >= 0.45 * sr:
            continue
        b, a = iirpeak(f / (sr / 2), Q=rng.uniform(4.0, 10.0))
        y += lfilter(b, a, src) * (1.0, 0.6, 0.3)[i]
    return y * _env(n, sr, rng.uniform(0.02, 0.08), rng.uniform(0.05, 0.2))


def consonant(dur: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """声の子音（サ行・シャ行・タ行に近い、帯域を限った雑音の短い音）。ハイハットと取り違えやすい音。"""
    n = max(int(dur * sr), 8)
    lo = rng.uniform(2500.0, 7000.0)
    hi = min(lo * rng.uniform(1.4, 2.5), 0.48 * sr)
    if hi <= lo * 1.05:
        lo = hi / 1.5
    b, a = butter(2, [lo / (sr / 2), hi / (sr / 2)], btype="band")
    t = np.arange(n) / sr
    return lfilter(b, a, rng.standard_normal(n)) * np.minimum(t / 0.004, 1.0) * np.exp(-t / rng.uniform(0.02, 0.09))


class _Notes:
    """同じ音（楽器・音程・長さ）を作り直さないための置き場。少しずつ違う音を 2 つ持って交互に使う。"""

    def __init__(self, sr: int, rng: np.random.Generator):
        self.sr, self.rng = sr, rng
        self.cache: dict[tuple, list[np.ndarray]] = {}

    def get(self, fn, note: int, dur: float) -> np.ndarray:
        key = (fn.__name__, note, round(dur, 2))
        if key not in self.cache:
            self.cache[key] = [fn(midi_hz(note), dur, self.sr, self.rng) for _ in range(2)]
        v = self.cache[key]
        return v[int(self.rng.integers(len(v)))]


def _add(out: np.ndarray, y: np.ndarray, start_sec: float, sr: int, gain: float = 1.0) -> None:
    a = int(start_sec * sr)
    if a >= out.size or a < 0:
        return
    b = min(a + y.size, out.size)
    out[a:b] += y[: b - a] * gain


def render_bleed(seconds: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """ドラム以外の楽器だけの音を `seconds` 秒ぶん作る（モノラル、ピーク 0.9 以下）。"""
    n = int(seconds * sr)
    bpm = rng.uniform(70.0, 170.0)
    beat = 60.0 / bpm
    scale = MAJOR if rng.random() < 0.5 else MINOR
    root = int(rng.integers(36, 48))  # ベースの音域の主音
    prog = PROGRESSIONS[int(rng.integers(len(PROGRESSIONS)))]
    swing = rng.uniform(0.0, 0.12) if rng.random() < 0.3 else 0.0
    notes = _Notes(sr, rng)

    def degree(d: int, octave: int = 0) -> int:
        return root + 12 * octave + scale[d % 7] + 12 * (d // 7)

    def chord(bar: int, octave: int) -> list[int]:
        d = prog[bar % len(prog)]
        return [degree(d + k, octave) for k in (0, 2, 4)]

    def at(bar: int, eighth: float) -> float:
        """小節と 8 分の位置 → 秒（裏拍を少し遅らせる揺れ付き）。"""
        late = swing * beat if int(eighth) % 2 == 1 else 0.0
        return (bar * 4 + eighth / 2.0) * beat + late + rng.normal(0.0, 0.004)

    n_bars = int(np.ceil(seconds / (4 * beat))) + 1
    names = ["bass", "guitar", "keys", "pad", "lead", "voice"]
    chosen = [nm for nm in names if rng.random() < 0.55]
    if not chosen:
        chosen = [names[int(rng.integers(len(names)))]]
    tracks: dict[str, np.ndarray] = {}
    for name in chosen:
        y = np.zeros(n)
        pattern = rng.random(8) < rng.uniform(0.35, 0.9)  # この楽器が鳴る 8 分の位置
        pattern[0] = True
        for bar in range(n_bars):
            ch = chord(bar, 0)
            if name == "bass":
                for e in np.flatnonzero(pattern):
                    note = ch[0] if rng.random() < 0.75 else ch[int(rng.integers(3))]
                    _add(y, notes.get(bass, note, min(beat * rng.choice([0.5, 1.0, 2.0]), 1.5)), at(bar, e), sr, rng.uniform(0.6, 1.0))
            elif name == "guitar":
                strings = [degree(prog[bar % len(prog)] + k, 1) for k in (0, 2, 4, 7, 9)]
                arpeggio = rng.random() < 0.4
                for e in np.flatnonzero(pattern):
                    if arpeggio:
                        for k in range(2):  # 16 分のアルペジオ
                            _add(y, notes.get(pluck, strings[int(rng.integers(len(strings)))], 1.2), at(bar, e) + k * beat / 4, sr, rng.uniform(0.5, 1.0))
                    else:
                        spread = rng.uniform(0.004, 0.025) * (1 if e % 2 == 0 else -1)  # ダウンとアップで弦の順が逆
                        for k, s in enumerate(strings):
                            _add(y, notes.get(pluck, s, 1.2), at(bar, e) + abs(spread) * (k if spread > 0 else len(strings) - 1 - k), sr, rng.uniform(0.5, 1.0))
            elif name == "keys":
                for e in np.flatnonzero(pattern[:8:2] if rng.random() < 0.5 else pattern):
                    for note in chord(bar, 1 if rng.random() < 0.7 else 2):
                        _add(y, notes.get(keys, note, 1.5), at(bar, e) + rng.uniform(0.0, 0.008), sr, rng.uniform(0.5, 1.0))
            elif name == "pad":
                for note in chord(bar, 1):
                    _add(y, notes.get(pad, note, 4 * beat + 0.3), bar * 4 * beat, sr, 0.5)
            elif name == "lead":
                d = prog[bar % len(prog)] + int(rng.integers(0, 5))
                for e in range(8):
                    if rng.random() < 0.5:
                        d += int(rng.integers(-2, 3))
                        _add(y, notes.get(lead, degree(d, 2), beat * rng.choice([0.5, 0.5, 1.0])), at(bar, e), sr, rng.uniform(0.6, 1.0))
            elif name == "voice":
                e = 0.0
                while e < 8:
                    length = float(rng.choice([1, 1, 2, 3, 4]))  # 8 分の数
                    if rng.random() < 0.7:
                        t0 = at(bar, e)
                        d = prog[bar % len(prog)] + int(rng.integers(0, 6))
                        if rng.random() < 0.6:  # 子音 → 母音
                            c = consonant(rng.uniform(0.04, 0.15), sr, rng)
                            _add(y, c, t0, sr, rng.uniform(0.3, 1.0))
                            t0 += min(c.size / sr * 0.5, 0.05)
                        _add(y, vowel(midi_hz(degree(d, 2)), length * beat / 2 * rng.uniform(0.7, 1.0), sr, rng), t0, sr, rng.uniform(0.6, 1.0))
                    e += length
        rms = np.sqrt(np.mean(y**2))
        if rms > 1e-6:
            y = y / rms
            if name in ("guitar", "lead", "keys") and rng.random() < 0.4:  # 歪み（倍音が増えて、中〜高域が埋まる）
                y = np.tanh(rng.uniform(2.0, 8.0) * y)
                y /= np.sqrt(np.mean(y**2)) + 1e-9
            tracks[name] = y * 10 ** (rng.uniform(-12.0, 0.0) / 20)
    out = sum(tracks.values()) if tracks else np.zeros(n)
    if rng.random() < 0.4:  # 部屋の響き
        from scipy.signal import fftconvolve

        rt = rng.uniform(0.3, 1.2)
        m = int(rt * sr)
        ir = rng.standard_normal(m) * np.exp(-6.9 * np.arange(m) / sr / rt)
        out = out + rng.uniform(0.1, 0.4) * fftconvolve(out, ir / (np.sqrt(np.sum(ir**2)) + 1e-9))[:n]
    peak = np.max(np.abs(out)) + 1e-9
    return (out / peak * 0.9).astype(np.float32)
