"""エフェクトの当てはめの最小版（計画書 3.5。合成による分析）。

固定の形のチェーン: 入力ゲイン → EQ（ローシェルフ 100 Hz、ピーキング、ハイシェルフ 5 kHz）→ リバーブ（指数減衰の雑音、RT60・高域の減衰・混ぜる量）。
コンプ・歪み・ゲートは Phase 0 では入れない（まず「効く」ことの確認）。

当てはめ: 「乾いた音にチェーンを掛けたもの」と「目標（ステム）」の多重解像度 STFT 距離を、
解析的な初期値から scipy の勾配を使わない最適化（Powell）で小さくする。
PyTorch の微分可能な版は、この形で効くと分かってから書く。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve, lfilter

from .metrics import multi_resolution_stft_distance


@dataclass
class ChainParams:
    gain_db: float = 0.0
    low_shelf_db: float = 0.0  # 100 Hz
    peak_hz: float = 1000.0
    peak_db: float = 0.0
    peak_q: float = 1.0
    high_shelf_db: float = 0.0  # 5 kHz
    reverb_rt60: float = 0.3  # 秒
    reverb_damp: float = 0.5  # 0（明るい）〜1（暗い）
    reverb_mix: float = 0.0  # 0〜1

    def to_dict(self) -> dict:
        return asdict(self)


# 最適化する項目と範囲（それ以外は固定）
_FIT_FIELDS: tuple[tuple[str, float, float], ...] = (
    ("gain_db", -24.0, 24.0),
    ("low_shelf_db", -18.0, 18.0),
    ("peak_hz", 200.0, 8000.0),  # 対数で扱う
    ("peak_db", -18.0, 18.0),
    ("high_shelf_db", -18.0, 18.0),
    ("reverb_rt60", 0.05, 3.0),  # 対数で扱う
    ("reverb_damp", 0.0, 1.0),
    ("reverb_mix", 0.0, 1.0),
)
_LOG_FIELDS = {"peak_hz", "reverb_rt60"}


# ---------------------------------------------------------------------------
# RBJ のバイクアッド
# ---------------------------------------------------------------------------

def _shelf(sr: float, f0: float, gain_db: float, high: bool) -> tuple[np.ndarray, np.ndarray]:
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / 2 * np.sqrt(2.0)  # S = 1
    sq = 2 * np.sqrt(A) * alpha
    if high:
        b = [A * ((A + 1) + (A - 1) * cw + sq), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sq)]
        a = [(A + 1) - (A - 1) * cw + sq, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sq]
    else:
        b = [A * ((A + 1) - (A - 1) * cw + sq), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sq)]
        a = [(A + 1) + (A - 1) * cw + sq, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sq]
    a = np.asarray(a, float)
    return np.asarray(b, float) / a[0], a / a[0]


def _peaking(sr: float, f0: float, gain_db: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2 * q)
    b = np.asarray([1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A])
    a = np.asarray([1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A])
    return b / a[0], a / a[0]


def _reverb_ir(sr: int, rt60: float, damp: float, seed: int = 12345) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(min(max(rt60, 0.05), 3.0) * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-6.9078 * t / rt60)  # -60 dB で rt60
    # 高域の減衰: 1次ローパス。damp=0 で 16 kHz、damp=1 で 1 kHz（対数）
    fc = 16000.0 * (1000.0 / 16000.0) ** damp
    a = np.exp(-2 * np.pi * fc / sr)
    ir = lfilter([1 - a], [1, -a], ir)
    ir /= np.sqrt(np.sum(ir**2)) + 1e-9
    return ir


def apply_chain(x: np.ndarray, sr: int, p: ChainParams) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64).ravel() * 10 ** (p.gain_db / 20)
    b, a = _shelf(sr, 100.0, p.low_shelf_db, high=False)
    y = lfilter(b, a, y)
    b, a = _peaking(sr, p.peak_hz, p.peak_db, p.peak_q)
    y = lfilter(b, a, y)
    b, a = _shelf(sr, 5000.0, p.high_shelf_db, high=True)
    y = lfilter(b, a, y)
    if p.reverb_mix > 0:
        wet = fftconvolve(y, _reverb_ir(sr, p.reverb_rt60, p.reverb_damp))[: y.size]
        y = (1 - p.reverb_mix) * y + p.reverb_mix * wet
    return y


# ---------------------------------------------------------------------------
# 初期値（解析的）
# ---------------------------------------------------------------------------

def _band_db(x: np.ndarray, sr: int, lo: float, hi: float) -> float:
    spec = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(x.size, 1 / sr)
    sel = (f >= lo) & (f < hi)
    return 20 * np.log10(np.sqrt(np.mean(spec[sel] ** 2)) + 1e-12)


def estimate_initial(dry: np.ndarray, wet: np.ndarray, sr: int) -> ChainParams:
    """帯域ごとの長期スペクトルの差から EQ とゲインを、目標の尾の減衰から RT60 を粗く決める。"""
    dry = np.asarray(dry, float).ravel()
    wet = np.asarray(wet, float).ravel()
    n = min(dry.size, wet.size)
    dry, wet = dry[:n], wet[:n]
    mid = _band_db(wet, sr, 300, 3000) - _band_db(dry, sr, 300, 3000)
    low = _band_db(wet, sr, 30, 150) - _band_db(dry, sr, 30, 150)
    high = _band_db(wet, sr, 6000, 16000) - _band_db(dry, sr, 6000, 16000)
    p = ChainParams(gain_db=float(np.clip(mid, -24, 24)),
                    low_shelf_db=float(np.clip(low - mid, -18, 18)),
                    high_shelf_db=float(np.clip(high - mid, -18, 18)))

    # RT60: 目標の最後の 1 秒の包絡（RMS、20 ms 窓）の減衰の傾き
    win = int(0.02 * sr)
    tail = wet[-int(min(1.0, n / sr) * sr):]
    rms = np.sqrt(np.convolve(tail**2, np.ones(win) / win, mode="valid")[::win] + 1e-12)
    db = 20 * np.log10(rms)
    if db.size >= 4 and (db.max() - db.min()) > 6:
        t = np.arange(db.size) * win / sr
        slope = np.polyfit(t, db, 1)[0]  # dB/秒（負）
        if slope < -3:
            p.reverb_rt60 = float(np.clip(-60.0 / slope, 0.05, 3.0))
            p.reverb_mix = 0.2
    return p


# ---------------------------------------------------------------------------
# 当てはめ
# ---------------------------------------------------------------------------

def _to_vec(p: ChainParams) -> np.ndarray:
    v = []
    for name, lo, hi in _FIT_FIELDS:
        x = getattr(p, name)
        if name in _LOG_FIELDS:
            x, lo, hi = np.log(x), np.log(lo), np.log(hi)
        v.append((x - lo) / (hi - lo))  # 0〜1 に正規化
    return np.asarray(v)


def _from_vec(v: np.ndarray, base: ChainParams) -> ChainParams:
    p = ChainParams(**base.to_dict())
    for (name, lo, hi), u in zip(_FIT_FIELDS, v):
        u = float(np.clip(u, 0.0, 1.0))
        if name in _LOG_FIELDS:
            x = np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))
        else:
            x = lo + u * (hi - lo)
        setattr(p, name, float(x))
    return p


def fit_chain(
    dry: np.ndarray,
    wet: np.ndarray,
    sr: int,
    init: ChainParams | None = None,
    max_seconds: float = 8.0,
    maxiter: int = 4,
    fft_sizes: tuple[int, ...] = (1024, 256),
) -> tuple[ChainParams, dict]:
    """`apply_chain(dry) ≈ wet` になるパラメータを探す。返り値: (パラメータ, 記録)。"""
    dry = np.asarray(dry, float).ravel()
    wet = np.asarray(wet, float).ravel()
    n = min(dry.size, wet.size, int(max_seconds * sr))
    dry, wet = dry[:n], wet[:n]
    init = init or estimate_initial(dry, wet, sr)

    def objective(v: np.ndarray) -> float:
        penalty = float(np.sum(np.clip(v - 1, 0, None) ** 2 + np.clip(-v, 0, None) ** 2))
        y = apply_chain(dry, sr, _from_vec(v, init))
        return multi_resolution_stft_distance(wet, y, fft_sizes)["total"] + 10 * penalty

    d_before = multi_resolution_stft_distance(wet, dry, fft_sizes)["total"]
    d_init = objective(_to_vec(init))
    res = minimize(objective, _to_vec(init), method="Powell",
                   options={"maxiter": maxiter, "xtol": 1e-3, "ftol": 1e-4, "disp": False})
    best = _from_vec(res.x, init)
    d_after = multi_resolution_stft_distance(wet, apply_chain(dry, sr, best), fft_sizes)["total"]
    return best, {"distance_before": d_before, "distance_init": d_init, "distance_after": d_after,
                  "n_eval": int(res.nfev), "init": init.to_dict()}
