"""対数メルスペクトログラム（numpy だけ。C++ への移植のしやすさを優先して librosa を使わない）。

採譜モデルの入力。設定は `FeatureConfig` に固定し、ONNX 側と同じ値を使う。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import stft

from .labels import Hit, to_coarse


@dataclass(frozen=True)
class FeatureConfig:
    sr: int = 22050
    n_fft: int = 1024
    hop: int = 220  # 約 10 ms
    n_mels: int = 64
    f_min: float = 30.0
    f_max: float = 11000.0

    @property
    def frame_rate(self) -> float:
        return self.sr / self.hop


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, float) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, float) / 2595.0) - 1.0)


def mel_filterbank(cfg: FeatureConfig) -> np.ndarray:
    """[n_mels, n_fft//2+1] の三角フィルタ。"""
    n_bins = cfg.n_fft // 2 + 1
    f = np.linspace(0, cfg.sr / 2, n_bins)
    m_pts = np.linspace(_hz_to_mel(cfg.f_min), _hz_to_mel(cfg.f_max), cfg.n_mels + 2)
    f_pts = _mel_to_hz(m_pts)
    fb = np.zeros((cfg.n_mels, n_bins))
    for i in range(cfg.n_mels):
        lo, c, hi = f_pts[i], f_pts[i + 1], f_pts[i + 2]
        up = (f - lo) / max(c - lo, 1e-9)
        down = (hi - f) / max(hi - c, 1e-9)
        fb[i] = np.clip(np.minimum(up, down), 0, None)
    return fb


def log_mel(x: np.ndarray, cfg: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """返り値: [frames, n_mels]（float32）。入力は cfg.sr のモノラル。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    _, _, z = stft(x, nperseg=cfg.n_fft, noverlap=cfg.n_fft - cfg.hop, window="hann", padded=True, boundary="zeros")
    power = np.abs(z) ** 2
    mel = mel_filterbank(cfg) @ power
    return np.log1p(1000.0 * mel).T.astype(np.float32)


def frame_targets(hits: list[Hit], n_frames: int, cfg: FeatureConfig, classes: tuple[str, ...]) -> np.ndarray:
    """打点を [frames, classes] の 0/1 にする。打点のフレームと次のフレームを 1 にする（±1 フレームのぼかし）。"""
    y = np.zeros((n_frames, len(classes)), dtype=np.float32)
    idx = {c: i for i, c in enumerate(classes)}
    for h in hits:
        c = to_coarse(h.cls)
        if c not in idx:
            continue
        # stft は boundary="zeros" でフレームの中心が t=0 から始まる
        fr = int(round(h.time * cfg.frame_rate))
        for d in (0, 1):
            if 0 <= fr + d < n_frames:
                y[fr + d, idx[c]] = 1.0
    return y


def pick_hits(prob: np.ndarray, cfg: FeatureConfig, classes: tuple[str, ...], threshold: float = 0.5, min_gap_sec: float = 0.05) -> list[Hit]:
    """[frames, classes] の確率から打点を拾う（しきい値 ＋ 局所最大 ＋ 最小間隔）。"""
    from scipy.ndimage import maximum_filter1d

    hits: list[Hit] = []
    gap = max(int(min_gap_sec * cfg.frame_rate), 1)
    for ci, c in enumerate(classes):
        p = prob[:, ci]
        local = p >= maximum_filter1d(p, size=2 * gap + 1, mode="nearest")
        cand = np.flatnonzero(local & (p > threshold))
        last = -10**9
        for i in cand:
            if i - last >= gap:
                hits.append(Hit(float(i / cfg.frame_rate), c, None))
                last = i
    hits.sort(key=lambda h: h.time)
    return hits
