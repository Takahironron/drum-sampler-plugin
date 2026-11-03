"""NMFD（非負値行列因子分解の畳み込み版）によるドラムのパート分け（計画書 3.3 の (2)）。

ドラムのステムの振幅スペクトログラム V [F, T] を、パートごとの時間周波数のひな形 W[k] [F, L] と
活性 H[k] [T] の畳み込みの和で近似する:  V[f, t] ≈ Σ_k Σ_τ W[k, f, τ] H[k, t − τ]。

採譜の結果（打点）を活性の初期値にし（打点の前後だけを大きく、他は小さく）、ひな形は打点の直後の
スペクトログラムの中央値で初期化する。KL ダイバージェンスの乗法更新（Smaragdis 2004）を回し、
Wiener フィルタで複素 STFT をパートに分けて逆変換する。パートの和は元に等しい（残差の成分を使えば、残差が残りを吸う）。

学習モデルを使わないので権利が明確で軽い。質の上限は分離ツール（SpectraLayers）に及ばない前提で、
製品の経路（採譜 → パート分け → 切り出し）が自前で通るかを見るための最小版。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import istft, stft

from .labels import Hit, to_coarse

PARTS: tuple[str, ...] = ("kick", "snare", "hihat", "cymbal")


@dataclass(frozen=True)
class NMFDConfig:
    n_fft: int = 1024
    hop: int = 256
    template_frames: int = 10  # ひな形の長さ（フレーム。256 ホップ・44.1 kHz で約 58 ms）
    iterations: int = 40
    onset_halfwidth: int = 2  # 活性の初期値で打点の前に許す幅（フレーム）
    off_onset: float = 1e-3  # 打点から離れたところの活性の初期値（最大に対する比）
    # 打点の後に活性を許す長さ（秒）。余韻の長い種類ほど長く。ひな形（短い）を繰り返して余韻を表す
    post_onset_sec: tuple[tuple[str, float], ...] = (("kick", 0.3), ("snare", 0.25), ("hihat", 0.12), ("tom", 0.4), ("cymbal", 1.5))
    residual: bool = True  # 打点に縛られない残差の成分を 1 つ足す（にじみ・残響を吸う）
    power: float = 1.0  # V = |X|^power（1 で振幅）
    mask_exponent: float = 2.0  # Wiener マスクの指数


def _stft(x: np.ndarray, cfg: NMFDConfig):
    f, t, z = stft(x, nperseg=cfg.n_fft, noverlap=cfg.n_fft - cfg.hop, window="hann", padded=True, boundary="zeros")
    return f, t, z


def _istft(z: np.ndarray, cfg: NMFDConfig, n: int) -> np.ndarray:
    _, y = istft(z, nperseg=cfg.n_fft, noverlap=cfg.n_fft - cfg.hop, window="hann", input_onesided=True, boundary=True)
    y = np.asarray(y, dtype=np.float64)
    if y.size < n:
        y = np.pad(y, (0, n - y.size))
    return y[:n]


def _approx(W: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Λ[f, t] = Σ_k Σ_τ W[k, f, τ] H[k, t − τ]"""
    K, F, L = W.shape
    T = H.shape[1]
    lam = np.zeros((F, T))
    for k in range(K):
        for tau in range(L):
            if tau == 0:
                lam += np.outer(W[k, :, 0], H[k])
            else:
                lam[:, tau:] += np.outer(W[k, :, tau], H[k, : T - tau])
    return lam


def spectral_prior(cls: str, freqs: np.ndarray) -> np.ndarray:
    """種類ごとの帯域の重み（ひな形の初期値に掛ける。0.05〜1）。キックは低域、ハイハットとシンバルは高域、スネアは中域。"""
    f = np.asarray(freqs, dtype=float)
    if cls == "kick":
        w = 1.0 / (1.0 + (f / 250.0) ** 2)
    elif cls == "snare":
        w = 1.0 / (1.0 + (f / 6000.0) ** 2) * (1.0 - 1.0 / (1.0 + (f / 120.0) ** 2))
    elif cls == "tom":
        w = 1.0 / (1.0 + (f / 1500.0) ** 2) * (1.0 - 1.0 / (1.0 + (f / 60.0) ** 2))
    elif cls == "hihat":
        w = 1.0 - 1.0 / (1.0 + (f / 3000.0) ** 2)
    elif cls == "cymbal":
        w = 1.0 - 1.0 / (1.0 + (f / 1500.0) ** 2)
    else:
        w = np.ones_like(f)
    return np.clip(w, 0.05, 1.0)


def initialize(V: np.ndarray, hits: list[Hit], frame_rate: float, cfg: NMFDConfig, parts: tuple[str, ...] = PARTS,
               freqs: np.ndarray | None = None):
    """活性 H を打点から、ひな形 W を打点直後のパッチの中央値から初期化する。

    ひな形は、他の種類と同時（±30 ms）でない打があればそれだけから作り（キックと同時のクラッシュなどの混入を避ける）、
    種類ごとの帯域の重みを掛ける。
    """
    F, T = V.shape
    K = len(parts) + (1 if cfg.residual else 0)
    L = cfg.template_frames
    H = np.full((K, T), cfg.off_onset)
    W = np.full((K, F, L), 1e-3)
    rng = np.random.default_rng(0)
    post = dict(cfg.post_onset_sec)
    all_times = sorted((h.time, to_coarse(h.cls)) for h in hits)
    for k, cls in enumerate(parts):
        own = sorted({h.time for h in hits if to_coarse(h.cls) == cls and 0 <= h.time * frame_rate < T})
        frames = [int(round(t * frame_rate)) for t in own]
        n_post = max(int(post.get(cls, 0.2) * frame_rate), cfg.onset_halfwidth)
        for fr in frames:
            lo, hi = max(fr - cfg.onset_halfwidth, 0), min(fr + n_post + 1, T)
            # 打点で 1、その後は直線に下げて off_onset へ（余韻を許しつつ、次の打までに消える形）
            ramp = np.linspace(1.0, cfg.off_onset, hi - lo)
            H[k, lo:hi] = np.maximum(H[k, lo:hi], ramp)
        isolated = [t for t in own if not any(c != cls and abs(t2 - t) <= 0.03 for t2, c in all_times)]
        use = isolated if len(isolated) >= 3 else own
        patches = [V[:, fr : fr + L] for fr in (int(round(t * frame_rate)) for t in use) if fr + L <= T]
        if patches:
            W[k] = np.median(np.stack(patches), axis=0) + 1e-6
        else:  # その種類の打が無ければ、平均的な形で薄く初期化
            W[k] = V.mean(axis=1, keepdims=True) * np.linspace(1.0, 0.2, L)[None, :] * 0.1 + 1e-6
        if freqs is not None:
            W[k] *= spectral_prior(cls, freqs)[:, None]
    if cfg.residual:
        H[-1] = 0.1 * rng.uniform(0.5, 1.5, T)
        W[-1] = V.mean(axis=1, keepdims=True) * np.ones((1, L)) / L + 1e-6
    # ひな形の大きさをそろえる（各ひな形の和を 1 に）
    W /= W.sum(axis=(1, 2), keepdims=True) + 1e-12
    H *= V.sum() / (K * max(T, 1))
    return W, H


def update(V: np.ndarray, W: np.ndarray, H: np.ndarray, cfg: NMFDConfig, H_mask: np.ndarray | None = None):
    """KL ダイバージェンスの乗法更新を 1 回。`H_mask` があれば活性をそれで抑える（打点の近くだけ許すなど）。"""
    K, F, L = W.shape
    T = H.shape[1]
    eps = 1e-10
    lam = _approx(W, H) + eps
    R = V / lam
    # H の更新
    numH = np.zeros_like(H)
    denH = np.zeros_like(H)
    for k in range(K):
        for tau in range(L):
            w = W[k, :, tau]
            numH[k, : T - tau] += w @ R[:, tau:]
            denH[k, : T - tau] += w.sum()
    H = H * (numH / (denH + eps))
    if H_mask is not None:
        H = H * H_mask
    lam = _approx(W, H) + eps
    R = V / lam
    # W の更新
    for k in range(K):
        for tau in range(L):
            h = H[k, : T - tau]
            numW = R[:, tau:] @ h
            denW = h.sum() + eps
            W[k, :, tau] = W[k, :, tau] * (numW / denW)
    W = np.maximum(W, 1e-12)
    W /= W.sum(axis=(1, 2), keepdims=True) + 1e-12  # 大きさは H に持たせる
    return W, H


def separate(x: np.ndarray, sr: int, hits: list[Hit], cfg: NMFDConfig | None = None, parts: tuple[str, ...] = PARTS,
             constrain_onsets: bool = True) -> dict[str, np.ndarray]:
    """ドラムのステムをパートに分ける。返り値: 種類 → 波形（元と同じ長さ）。`residual` があれば "residual" も返す。

    `constrain_onsets`: 打点の前後以外の活性を小さく抑え続ける（採譜の結果に忠実。にじみは残差の成分へ）。
    """
    cfg = cfg or NMFDConfig()
    x = np.asarray(x, dtype=np.float64).ravel()
    freqs, _, X = _stft(x, cfg)
    freqs = freqs * sr  # scipy は fs=1 のとき正規化周波数を返す
    V = np.abs(X) ** cfg.power + 1e-8
    frame_rate = sr / cfg.hop
    W, H = initialize(V, hits, frame_rate, cfg, parts, freqs)
    H_mask = None
    if constrain_onsets:
        # 初期の活性（打点からの距離で 1 → off_onset）をそのままマスクにする
        H_mask = np.ones_like(H)
        scale = H[: len(parts)].max(axis=1, keepdims=True) + 1e-12
        H_mask[: len(parts)] = np.clip(H[: len(parts)] / scale, cfg.off_onset, 1.0)
    for _ in range(cfg.iterations):
        W, H = update(V, W, H, cfg, H_mask)
    # Wiener マスクで複素 STFT を分ける
    comps = np.stack([_approx(W[k : k + 1], H[k : k + 1]) ** cfg.mask_exponent for k in range(W.shape[0])])
    total = comps.sum(axis=0)
    tiny = total < 1e-20
    masks = comps / np.where(tiny, 1.0, total)[None]
    masks[:, tiny] = 1.0 / comps.shape[0]  # どの成分も無いところは等分（マスクの和を常に 1 にし、パートの和を元に等しくする）
    out: dict[str, np.ndarray] = {}
    names = list(parts) + (["residual"] if cfg.residual else [])
    for name, m in zip(names, masks):
        out[name] = _istft(X * m, cfg, x.size)
    return out


def snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    """SNR（dB）。ref と est は同じ長さ。"""
    n = min(ref.size, est.size)
    r, e = ref[:n], est[:n]
    return float(10.0 * np.log10((np.sum(r**2) + 1e-12) / (np.sum((r - e) ** 2) + 1e-12)))
