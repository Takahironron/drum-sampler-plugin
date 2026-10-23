"""採点の指標。

- 採譜の F 値: `mir_eval.onset` を種類ごとに使う。時刻の許容誤差は ±50 ms（既定）。
- 多重解像度 STFT 距離: 複数の窓長の STFT について、スペクトル収束と対数振幅の L1 の和を平均する
  （Yamamoto ら 2020 の Parallel WaveGAN、auraloss の MultiResolutionSTFTLoss と同じ形）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.signal import stft

from .labels import COARSE_CLASSES, M0_CLASSES, Hit, to_coarse

ONSET_TOLERANCE_SEC = 0.05


@dataclass(frozen=True)
class ClassScore:
    precision: float
    recall: float
    f_measure: float
    n_ref: int
    n_est: int


def _times_by_class(hits: Iterable[Hit], classes: Sequence[str]) -> dict[str, np.ndarray]:
    buckets: dict[str, list[float]] = {c: [] for c in classes}
    for h in hits:
        c = to_coarse(h.cls)
        if c in buckets:
            buckets[c].append(h.time)
    return {c: np.asarray(sorted(v), dtype=float) for c, v in buckets.items()}


def transcription_scores(
    reference: Iterable[Hit],
    estimated: Iterable[Hit],
    classes: Sequence[str] = COARSE_CLASSES,
    tolerance: float = ONSET_TOLERANCE_SEC,
) -> dict[str, ClassScore]:
    """種類ごとの P / R / F。正解も推定も無い種類は P=R=F=1、片方だけ無い種類は 0。"""
    import mir_eval

    ref = _times_by_class(reference, classes)
    est = _times_by_class(estimated, classes)
    out: dict[str, ClassScore] = {}
    for c in classes:
        n_ref, n_est = int(ref[c].size), int(est[c].size)
        if n_ref == 0 and n_est == 0:
            p = r = f = 1.0
        elif n_ref == 0 or n_est == 0:
            p = r = f = 0.0
        else:
            f, p, r = mir_eval.onset.f_measure(ref[c], est[c], window=tolerance)
        out[c] = ClassScore(float(p), float(r), float(f), n_ref, n_est)
    return out


def m0_transcription_score(
    reference: Iterable[Hit],
    estimated: Iterable[Hit],
    tolerance: float = ONSET_TOLERANCE_SEC,
) -> float:
    """M0-1 の数字: キック・スネア・ハイハットの F 値の単純平均。"""
    scores = transcription_scores(reference, estimated, M0_CLASSES, tolerance)
    return float(np.mean([scores[c].f_measure for c in M0_CLASSES]))


# ---------------------------------------------------------------------------
# 多重解像度 STFT 距離
# ---------------------------------------------------------------------------

DEFAULT_FFT_SIZES: tuple[int, ...] = (2048, 1024, 512, 256, 128)


def _magnitude(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    _, _, z = stft(x, nperseg=n_fft, noverlap=n_fft - hop, window="hann", padded=True, boundary="zeros")
    return np.abs(z)


def multi_resolution_stft_distance(
    x: np.ndarray,
    y: np.ndarray,
    fft_sizes: Sequence[int] = DEFAULT_FFT_SIZES,
    eps: float = 1e-7,
) -> dict[str, float]:
    """`x`（基準、例: 元のステム）と `y`（比較、例: 再現）の距離。

    返り値: `{"total", "spectral_convergence", "log_magnitude"}`。小さいほど近い。
    長さが違うときは短い方に合わせる。モノラル（1次元）を前提にする。
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = min(x.size, y.size)
    if n == 0:
        raise ValueError("空の信号")
    x, y = x[:n], y[:n]

    sc_list, lm_list = [], []
    for n_fft in fft_sizes:
        hop = n_fft // 4
        mx = _magnitude(x, n_fft, hop)
        my = _magnitude(y, n_fft, hop)
        sc = np.linalg.norm(mx - my) / (np.linalg.norm(mx) + eps)
        lm = np.mean(np.abs(np.log(mx + eps) - np.log(my + eps)))
        sc_list.append(float(sc))
        lm_list.append(float(lm))
    sc_mean = float(np.mean(sc_list))
    lm_mean = float(np.mean(lm_list))
    return {
        "total": sc_mean + lm_mean,
        "spectral_convergence": sc_mean,
        "log_magnitude": lm_mean,
    }
