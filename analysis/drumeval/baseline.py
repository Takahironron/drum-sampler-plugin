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


# パート別ステムの打点を拾うときに見る帯域（Hz。上限 None は上まで）。その種類の音が主に鳴る帯域だけを見ると、
# 他の種類の漏れ（キックのステムに入ったスネアやハイハット）で誤検出しにくい
PART_BANDS_HZ: dict[str, tuple[float, float | None]] = {
    "kick": (25.0, 180.0), "snare": (150.0, None), "hihat": (3000.0, None), "tom": (50.0, 600.0), "cymbal": (2000.0, None),
}


def onsets_from_part(x: np.ndarray, sr: int, cls: str | None = None, rise_db: float = 6.0, rel_floor_db: float = -30.0,
                     min_interval_sec: float = 0.05, win_sec: float = 0.02, look_sec: float = 0.06, return_levels: bool = False):
    """パート別に分離された音（キックだけ、スネアだけ…）から打点の時刻（秒）を拾う。分離ツールのパート別ステムを
    「疑似的な正解」にするために使う（分離の誤りはそのまま残る）。

    - 包絡は、直前 `win_sec` の二乗平均（4 ms ごと）。`cls` があればその種類の帯域（`PART_BANDS_HZ`）だけを見る。
      打点より前の音を含まない（ヒルベルト変換の包絡は打点の前に裾を引き、無音の後の打を 80 ms 早く拾った）。
    - 包絡が、**その窓の前 `look_sec` の平均**を `rise_db` 以上超えた瞬間を打点にする。直前の値や最小と比べると、
      余韻の揺れ（キックの低い音のうねり）を立ち上がりと取り違える。平均と比べれば、揺れは平均を大きくは超えない。
    - 打の大きさ（打点から 40 ms の包絡の最大）が、大きい打（上位 5%）より `rel_floor_db` 以上小さいものは捨てる
      （余韻の揺れ、他の種類の漏れ）。ごく弱いゴーストノートも落ちる。
    - `return_levels` なら (時刻, 大きさ dB) を返す。大きさは大きい打（上位 5%）を 0 dB とした相対値。

    2026-11-15 に作り直した。それまでは「約 12 ms 前より 6 dB 大きい」だけで拾っていて、本当のキックの単独録音（StemGMD の 12 曲）で
    正解 742 に対して 2226 個を検出していた（余韻の途中と、他の太鼓を叩いたときの漏れ）。作り直した後は 727 個
    （適合率 0.99、再現率 0.97。スネア 1.00／0.81、ハイハット 1.00／0.92。他のパートを −30 dB で混ぜても適合率 0.97 以上）。
    """
    from scipy.signal import butter, sosfilt

    empty = (np.zeros(0), np.zeros(0)) if return_levels else np.zeros(0)
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < int(0.1 * sr) or np.max(np.abs(x)) < 1e-6:
        return empty
    if cls in PART_BANDS_HZ:
        lo, hi = PART_BANDS_HZ[cls]
        nyq = sr / 2
        if hi is None or hi >= 0.95 * nyq:
            sos = butter(2, min(lo / nyq, 0.9), btype="high", output="sos")
        else:
            sos = butter(2, [lo / nyq, hi / nyq], btype="band", output="sos")
        x = sosfilt(sos, x)  # 因果的に掛ける（打点の前に響かせない）
    hop = max(int(0.004 * sr), 1)
    n = x.size // hop
    block = np.sum(x[: n * hop].reshape(n, hop) ** 2, axis=1) / hop  # ホップごとの平均パワー
    cum = np.concatenate([[0.0], np.cumsum(block)])
    w = max(int(round(win_sec * sr / hop)), 1)
    look = max(int(round(look_sec * sr / hop)), 1)
    end = np.arange(1, n + 1)  # フレーム i の包絡は、ブロック (i − w, i] の平均
    a0 = np.maximum(end - w, 0)
    b0 = np.maximum(a0 - look, 0)
    cur = (cum[end] - cum[a0]) / np.maximum(end - a0, 1) + 1e-14
    prev = (cum[a0] - cum[b0]) / np.maximum(a0 - b0, 1) + 1e-14  # その窓の前 look ブロックの平均
    e = 10.0 * np.log10(cur)
    above = (e - 10.0 * np.log10(prev)) > rise_db
    above[:w] = False
    starts = np.flatnonzero(above & ~np.concatenate([[False], above[:-1]]))
    if starts.size == 0:
        return empty
    frame_rate = sr / hop
    hold = max(int(0.04 * frame_rate), 1)
    peak = np.array([e[i : i + hold + 1].max() for i in starts])
    ref_peak = np.percentile(peak, 95)
    gap = max(int(min_interval_sec * frame_rate), 1)
    kept: list[int] = []
    levels: list[float] = []
    for i, pk in zip(starts, peak):
        if pk >= ref_peak + rel_floor_db and (not kept or i - kept[-1] >= gap):
            kept.append(int(i))
            levels.append(float(pk - ref_peak))
    times = np.asarray([(i + 0.5) * hop / sr for i in kept])  # 超えたブロックの中ほど
    return (times, np.asarray(levels)) if return_levels else times
