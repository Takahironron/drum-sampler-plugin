"""学習モデル無しの採譜の基準値（スペクトル・フラックス ＋ 帯域の比による分類）。

目的は、公開モデルや自前の CRNN（3〜4週目）が「これより良い」ことを確かめる下限を作ること。
精度は期待しない。

手順:
1. STFT（窓 1024、ホップ 256）の対数振幅から、3つの帯域（低: 20〜100 Hz、中: 150〜2 kHz、高: 5 kHz〜）の
   スペクトル・フラックス（正の差分の和を帯域の bin 数で割ったもの）を出す。
2. 低＋中のフラックスで山（打点）を拾い、打点直後（約 30 ms）の振幅スペクトルの
   低（40〜100 Hz）／中（150〜2 kHz）のエネルギー比が `kick_energy_ratio` 以上なら kick、それ以外は snare。
3. 高のフラックスで別に山を拾い、hihat にする（スネアやキックと同時でもよい）。
   山は適応しきい値（移動中央値 ＋ 定数）を超えるものだけ。山の間隔は最小 50 ms。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter
from scipy.signal import stft

from .labels import Hit


@dataclass(frozen=True)
class BaselineConfig:
    n_fft: int = 1024
    hop: int = 256
    min_interval_sec: float = 0.05
    median_sec: float = 0.3
    threshold_lowmid: float = 0.3  # 低＋中の山のしきい値（移動中央値に足す定数）
    threshold_high: float = 0.06  # 高の山のしきい値
    kick_energy_ratio: float = 8.0  # 打点直後の 低／中 のエネルギー比がこれ以上なら kick、未満なら snare
    kick_energy_band_hz: tuple[float, float] = (40.0, 100.0)
    energy_frames: int = 5  # 打点直後に見るフレーム数（ホップ 256 で約 30 ms）
    bands_hz: tuple[tuple[float, float], ...] = ((20.0, 100.0), (150.0, 2000.0), (5000.0, 20000.0))


def spectrogram(x: np.ndarray, sr: int, cfg: BaselineConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返り値: (周波数 [bin], 時刻 [frame], 振幅 [bin, frame])"""
    f, t, z = stft(x, fs=sr, nperseg=cfg.n_fft, noverlap=cfg.n_fft - cfg.hop, window="hann", padded=True, boundary="zeros")
    return f, t, np.abs(z)


def band_flux(x: np.ndarray, sr: int, cfg: BaselineConfig) -> tuple[np.ndarray, np.ndarray]:
    """返り値: (時刻 [frame], フラックス [band, frame])"""
    f, t, mag = spectrogram(x, sr, cfg)
    return t, _band_flux_from_mag(f, mag, cfg)


def _band_flux_from_mag(f: np.ndarray, mag: np.ndarray, cfg: BaselineConfig) -> np.ndarray:
    mag = np.log1p(100.0 * mag)
    diff = np.diff(mag, axis=1, prepend=mag[:, :1])
    pos = np.maximum(diff, 0.0)
    flux = []
    for lo, hi in cfg.bands_hz:
        sel = (f >= lo) & (f < hi)
        flux.append(pos[sel].sum(axis=0) / max(sel.sum(), 1))
    return np.asarray(flux)


def pick_peaks(flux: np.ndarray, frame_rate: float, cfg: BaselineConfig, threshold: float) -> np.ndarray:
    """1本のフラックスから山のフレーム番号を返す。"""
    med_len = max(int(cfg.median_sec * frame_rate) | 1, 3)
    thresh = median_filter(flux, size=med_len, mode="nearest") + threshold
    min_dist = max(int(cfg.min_interval_sec * frame_rate), 1)
    local_max = flux >= maximum_filter1d(flux, size=2 * min_dist + 1, mode="nearest")
    cand = np.flatnonzero(local_max & (flux > thresh))
    # 最小間隔を守る（先に見つけた山を優先）
    kept: list[int] = []
    for i in cand:
        if not kept or i - kept[-1] >= min_dist:
            kept.append(int(i))
    return np.asarray(kept, dtype=int)


def transcribe(x: np.ndarray, sr: int, cfg: BaselineConfig | None = None) -> list[Hit]:
    cfg = cfg or BaselineConfig()
    x = np.asarray(x, dtype=np.float64).ravel()
    f, t, mag = spectrogram(x, sr, cfg)
    low, mid, high = _band_flux_from_mag(f, mag, cfg)
    frame_rate = sr / cfg.hop
    sel_low = (f >= cfg.kick_energy_band_hz[0]) & (f < cfg.kick_energy_band_hz[1])
    sel_mid = (f >= cfg.bands_hz[1][0]) & (f < cfg.bands_hz[1][1])
    # 打の立ち上がりは差分が出たフレームの1つ前から始まるので半フレーム戻す
    offset = 0.5 / frame_rate
    hits: list[Hit] = []

    for i in pick_peaks(low + mid, frame_rate, cfg, cfg.threshold_lowmid):
        e_low = mag[sel_low, i : i + cfg.energy_frames].mean()
        e_mid = mag[sel_mid, i : i + cfg.energy_frames].mean()
        cls = "kick" if e_low >= cfg.kick_energy_ratio * e_mid else "snare"
        hits.append(Hit(float(t[i]) - offset, cls, None))

    for i in pick_peaks(high, frame_rate, cfg, cfg.threshold_high):
        hits.append(Hit(float(t[i]) - offset, "hihat", None))

    hits.sort(key=lambda h: h.time)
    return hits


def onsets_from_part(x: np.ndarray, sr: int, cfg: BaselineConfig | None = None, rise_db: float = 6.0) -> np.ndarray:
    """パート別に分離された音（キックだけ、スネアだけ…）から打点の時刻（秒）を拾う。

    1 パートの音は打点で立ち上がり、あとは減衰するだけなので、フレームごとのエネルギー（dB）が
    直前（約 20 ms 前）より `rise_db` 以上大きくなった局所最大を打点とする。スペクトル・フラックスだと
    キックのピッチの下降（低い bin が後から増える）で誤検出するため、エネルギーの増分を使う。
    分離ツールが出すパート別ステムを「疑似的な正解」にするために使う（分離の誤りはそのまま残る）。
    """
    from scipy.ndimage import maximum_filter1d

    cfg = cfg or BaselineConfig()
    x = np.asarray(x, dtype=np.float64).ravel()
    peak = np.max(np.abs(x)) + 1e-9
    x = x / peak
    _, t, mag = spectrogram(x, sr, cfg)
    energy_db = 10.0 * np.log10(np.sum(mag**2, axis=0) + 1e-8)
    look = 2  # 約 20 ms 前と比べる
    prev = np.concatenate([np.full(look, energy_db[0]), energy_db[:-look]])
    rise = np.maximum(energy_db - prev, 0.0)
    frame_rate = sr / cfg.hop
    gap = max(int(cfg.min_interval_sec * frame_rate), 1)
    local = rise >= maximum_filter1d(rise, size=2 * gap + 1, mode="nearest")
    floor = np.percentile(energy_db, 10) + 10.0  # 無音に近い部分の揺れは打点にしない
    cand = np.flatnonzero(local & (rise > rise_db) & (energy_db > floor))
    kept: list[int] = []
    for i in cand:
        if not kept or i - kept[-1] >= gap:
            kept.append(int(i))
    # 立ち上がりは増分が出たフレームの 1 つ前から始まる
    return np.asarray([float(t[i]) - 0.5 / frame_rate for i in kept])
