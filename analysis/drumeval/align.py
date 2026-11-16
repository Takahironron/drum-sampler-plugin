"""正解の打点（MIDI）と音の時刻のずれを測る。

E-GMD の配布物では、正解の MIDI と音がファイルごとに一定量ずれているものがある（2026-11-16 に標本で確認。−73〜+62 ms）。
ずれたままだと、採譜モデルは数フレームずれたラベルで学習し、採点は正しい検出を誤りと数え（許容 ±50 ms）、
パート分け（NMFD）は打点の窓が音に合わない。ファイルごとに 1 つの値を測って、打点の時刻に足す。
"""

from __future__ import annotations

import numpy as np

from .labels import Hit


def onset_strength(x: np.ndarray, sr: int, hop_sec: float = 0.00145) -> tuple[np.ndarray, float]:
    """音の立ち上がりの強さ（対数で圧縮した振幅スペクトルの、増えた分の和）。返り値: (フレームごとの値, 1 秒あたりのフレーム数)。

    フレームの間隔は約 1.5 ms（ずれを数 ms の精度で測るため）。フレーム k の値は、時刻 k × hop の前後の窓から計算する。
    """
    from scipy.signal import stft

    x = np.asarray(x, dtype=np.float64).ravel()
    hop = max(int(round(hop_sec * sr)), 1)
    n_fft = 8 * hop
    _, _, z = stft(x, nperseg=n_fft, noverlap=n_fft - hop, window="hann", boundary="zeros", padded=True)
    mag = np.log1p(100.0 * np.abs(z) / (np.max(np.abs(z)) + 1e-12))  # 小さい打（ハイハット）も効くように圧縮する
    flux = np.maximum(mag[:, 1:] - mag[:, :-1], 0.0).sum(axis=0)
    return np.concatenate([[0.0], flux]), sr / hop


MIN_SHARPNESS = 1.5  # これ未満の確かさのずれは使わない（山がはっきりしない。67 本の標本では最小 1.7）


def label_offset(x: np.ndarray, sr: int, hits: list[Hit], max_lag_sec: float = 0.25) -> tuple[float, float]:
    """音が打点よりどれだけ遅れているか（秒。正なら音が遅い）と、その確かさ（相互相関の山の高さ ÷ 中央値）を返す。

    音の立ち上がりの強さと、打点に 1 を立てた列との相互相関を、±`max_lag_sec` の範囲で最大にするずれ。
    打点の時刻に返り値を足すと、音に合う。打が少ない（5 未満）ときは (0, 0)。
    E-GMD の標本 67 本では −73〜+119 ms。独立な測り方（キックとスネアの打点から最も近い音の立ち上がりまでの差の中央値）と、
    66 本で 6 ms 以内に一致した。
    """
    if len(hits) < 5:
        return 0.0, 0.0
    if sr >= 32000:  # 半分の sr で測る（結果の差は 0.6 ms 以内、時間は 1／3）
        from scipy.signal import resample_poly

        x, sr = resample_poly(np.asarray(x, dtype=np.float64).ravel(), 1, 2), sr // 2
    flux, rate = onset_strength(x, sr)
    train = np.zeros_like(flux)
    for h in hits:
        k = int(round(h.time * rate))
        if 0 <= k < train.size:
            train[k] += 1.0
    max_lag = int(max_lag_sec * rate)
    lags = np.arange(-max_lag, max_lag + 1)
    cc = np.empty(lags.size)
    n = flux.size
    for i, lag in enumerate(lags):
        cc[i] = float(np.dot(flux[max(lag, 0) : n + min(lag, 0)], train[max(-lag, 0) : n + min(-lag, 0)]))
    # 立ち上がりの強さの山は幅を持つので、少しならしてから山を取る（±2 フレーム ≒ 3 ms）
    smooth = np.convolve(cc, np.ones(5) / 5.0, mode="same")
    i = int(np.argmax(smooth))
    frac = 0.0
    if 0 < i < smooth.size - 1:  # 放物線で山の位置を 1 フレームより細かく
        a, b, c = smooth[i - 1], smooth[i], smooth[i + 1]
        if a - 2 * b + c < 0:
            frac = 0.5 * (a - c) / (a - 2 * b + c)
    sharpness = float(smooth[i] / (np.median(smooth) + 1e-12))
    return float((lags[i] + frac) / rate), sharpness


def shift_hits(hits: list[Hit], offset_sec: float) -> list[Hit]:
    """打点の時刻に `offset_sec` を足す（0 未満になる打は 0 に寄せる）。"""
    return [Hit(max(h.time + offset_sec, 0.0), h.cls, h.velocity) for h in hits]
