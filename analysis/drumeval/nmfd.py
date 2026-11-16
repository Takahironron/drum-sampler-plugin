"""NMFD（非負値行列因子分解の畳み込み版）によるドラムのパート分け（計画書 3.3 の (2)）。

ドラムのステムの振幅スペクトログラム V [F, T] を、パートごとの時間周波数のひな形 W[k] [F, L] と
活性 H[k] [T] の畳み込みの和で近似する:  V[f, t] ≈ Σ_k Σ_τ W[k, f, τ] H[k, t − τ]。

採譜の結果（打点）で活性を縛る。各打に「窓」（打点の少し前から、同じ種類の次の打の手前まで。種類ごとの上限あり）を作り、
活性は窓の中だけ許す。窓の中は自由なので、余韻（シンバル、残響）も、同じ種類の取りこぼした打も、そのパートが受ける。
ひな形は打点の直後のスペクトログラムの中央値で初期化する。KL ダイバージェンスの乗法更新（Smaragdis 2004）を回し、
Wiener フィルタで複素 STFT をパートに分けて逆変換する。パートと残差の和は元に等しい。

残差の成分（打点に縛られない 1 成分）は、どのパートでも説明できない音を受ける: 窓の外の取りこぼした打、打点の無い種類の音
（パーカッション）。ドラムで説明できる音まで残差が取らないように、残差の活性には軽い罰を掛ける。
ハイハットとシンバルのひな形には低域を持たせない（取りこぼしたキックがこれらのパートへ流れ込むのを防ぐ）。

限界: ステムに残った他の楽器のにじみは取り除けない。罰を掛けると、にじみの大半はスペクトルの近いパートに入る（罰が無いと半分ほどは
残差に行くが、ドラム自身の残響や余韻も残差に行って音が痩せる）。残差の成分を増やしても変わらなかった。にじみはステムの分離の側の仕事。

学習モデルを使わないので権利が明確で軽い。設定の根拠（E-GMD と StemGMD の標本での比較）は
`docs/phase0/結果/2026-11-15_段階1の聞き比べ_と_NMFDの修正.md`。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import istft, stft

from .labels import Hit, to_coarse

PARTS: tuple[str, ...] = ("kick", "snare", "hihat", "cymbal")
ALL_PARTS: tuple[str, ...] = ("kick", "snare", "hihat", "tom", "cymbal")


def parts_for_hits(hits: list[Hit], min_tom_hits: int = 4) -> tuple[str, ...]:
    """打点に合わせたパートの並び。タムの打が `min_tom_hits` 以上あればタムのパートも作る
    （無いとタムの音は残差か他のパートに入る。少数の誤検出だけでタムのパートを作らないための下限）。"""
    n_tom = sum(1 for h in hits if to_coarse(h.cls) == "tom")
    return ALL_PARTS if n_tom >= min_tom_hits else PARTS


@dataclass(frozen=True)
class NMFDConfig:
    n_fft: int = 1024
    hop: int = 256
    template_frames: int = 20  # ひな形の長さ（フレーム。約 0.23 秒）
    iterations: int = 40
    onset_halfwidth: int = 2  # 窓を打点の何フレーム前から開けるか
    off_onset: float = 1e-3  # 窓の外の活性に掛ける係数（ほぼ 0）
    # 窓の終わり: "to_next" は同じ種類の次の打の手前まで（上限 max_window_sec）。"fixed" は打点の後 post_onset_sec（従来）
    window: str = "to_next"
    max_window_sec: tuple[tuple[str, float], ...] = (("kick", 1.0), ("snare", 1.5), ("hihat", 1.0), ("tom", 1.5), ("cymbal", 4.0))
    post_onset_sec: tuple[tuple[str, float], ...] = (("kick", 0.3), ("snare", 0.25), ("hihat", 0.12), ("tom", 0.4), ("cymbal", 1.5))
    # 活性の縛り方: "support" は窓の中は自由・外は off_onset。"ramp" は従来（打点からの距離で 1 → off_onset に下がる係数を、
    # 更新のたびに活性へ掛ける。係数が累積して打点の後の音がパートから消えるので、比較用にだけ残す。`LEGACY`）
    constraint: str = "support"
    residual: bool = True  # 打点に縛られない残差の成分を 1 つ足す
    # 残差の活性への罰（疎にする項。更新の分母に 1 + この値 を掛ける）。0 だと、ドラムで説明できる音（残響、余韻）も残差が取る
    residual_penalty: float = 0.5
    # ひな形に持たせない低域（種類, この周波数 Hz 未満は 0）
    template_low_cut_hz: tuple[tuple[str, float], ...] = (("hihat", 300.0), ("cymbal", 200.0))
    power: float = 1.0  # V = |X|^power（1 で振幅）
    mask_exponent: float = 2.0  # Wiener マスクの指数


# 2026-11-15 より前の設定（聞き比べで「全体的にこもる」になった版）。比較用: `NMFDConfig(**LEGACY)`、`run_eval.py --nmfd-legacy`
LEGACY: dict = {"template_frames": 10, "window": "fixed", "constraint": "ramp", "residual_penalty": 0.0, "template_low_cut_hz": ()}


def config_for(sr: int, **overrides) -> NMFDConfig:
    """サンプリング周波数に合わせた設定（44.1 kHz 以上は窓 2048・ホップ 512、それ未満は 1024・256。フレームの間隔はどちらも約 12 ms）。"""
    base = {"n_fft": 2048, "hop": 512} if sr >= 40000 else {"n_fft": 1024, "hop": 256}
    return NMFDConfig(**{**base, **overrides})


def _stft(x: np.ndarray, cfg: NMFDConfig):
    f, t, z = stft(x, nperseg=cfg.n_fft, noverlap=cfg.n_fft - cfg.hop, window="hann", padded=True, boundary="zeros")
    return f, t, z


def _istft(z: np.ndarray, cfg: NMFDConfig, n: int) -> np.ndarray:
    _, y = istft(z, nperseg=cfg.n_fft, noverlap=cfg.n_fft - cfg.hop, window="hann", input_onesided=True, boundary=True)
    y = np.asarray(y, dtype=np.float64)
    if y.size < n:
        y = np.pad(y, (0, n - y.size))
    return y[:n]


def _shifted(H: np.ndarray, L: int) -> np.ndarray:
    """Hs[k, τ, t] = H[k, t − τ]（t < τ は 0）。形は [K, L, T]。"""
    K, T = H.shape
    Hs = np.zeros((K, L, T))
    for tau in range(L):
        Hs[:, tau, tau:] = H[:, : T - tau]
    return Hs


def _approx(W: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Λ[f, t] = Σ_k Σ_τ W[k, f, τ] H[k, t − τ]。[F, K·L] と [K·L, T] の行列積 1 回で計算する。"""
    K, F, L = W.shape
    return W.transpose(1, 0, 2).reshape(F, K * L) @ _shifted(H, L).reshape(K * L, H.shape[1])


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


def onset_windows(hits: list[Hit], frame_rate: float, T: int, cfg: NMFDConfig, parts: tuple[str, ...] = PARTS) -> list[list[tuple[int, int, int]]]:
    """種類ごとに、各打の窓 (始まり, 打点, 終わり) をフレームで返す。活性はこの窓の中だけ許す。

    `cfg.window == "to_next"`: 同じ種類の次の打の手前まで（上限 `max_window_sec`）。`"fixed"`: 打点の後 `post_onset_sec`。
    """
    post = dict(cfg.post_onset_sec)
    longest = dict(cfg.max_window_sec)
    out: list[list[tuple[int, int, int]]] = []
    for cls in parts:
        frames = sorted({int(round(h.time * frame_rate)) for h in hits if to_coarse(h.cls) == cls and 0 <= h.time * frame_rate < T})
        wins = []
        for i, fr in enumerate(frames):
            lo = max(fr - cfg.onset_halfwidth, 0)
            if cfg.window == "to_next":
                hi = min(fr + max(int(longest.get(cls, 1.0) * frame_rate), cfg.onset_halfwidth) + 1, T)
                if i + 1 < len(frames):
                    hi = min(hi, max(frames[i + 1] - cfg.onset_halfwidth, fr + 1))
            else:
                hi = min(fr + max(int(post.get(cls, 0.2) * frame_rate), cfg.onset_halfwidth) + 1, T)
            wins.append((lo, fr, hi))
        out.append(wins)
    return out


def initialize(V: np.ndarray, hits: list[Hit], frame_rate: float, cfg: NMFDConfig, parts: tuple[str, ...] = PARTS,
               freqs: np.ndarray | None = None):
    """活性 H を打点から、ひな形 W を打点直後のパッチの中央値から初期化する。返り値: (W, H, 窓)。

    ひな形は、他の種類と同時（±30 ms）でない打があればそれだけから作り（キックと同時のクラッシュなどの混入を避ける）、
    種類ごとの帯域の重みを掛ける。
    """
    F, T = V.shape
    K = len(parts) + (1 if cfg.residual else 0)
    L = cfg.template_frames
    H = np.full((K, T), cfg.off_onset)
    W = np.full((K, F, L), 1e-3)
    rng = np.random.default_rng(0)
    windows = onset_windows(hits, frame_rate, T, cfg, parts)
    all_times = sorted((h.time, to_coarse(h.cls)) for h in hits)
    for k, cls in enumerate(parts):
        own = sorted({h.time for h in hits if to_coarse(h.cls) == cls and 0 <= h.time * frame_rate < T})
        for lo, fr, hi in windows[k]:
            if cfg.constraint == "ramp":
                shape = np.linspace(1.0, cfg.off_onset, hi - lo)  # 打点で 1、その後は直線に下げる（従来）
            else:
                # 窓の頭から立ち上がり（3 フレーム）まで 1、その後は指数で下げる（形は初期値。後は更新が決める）
                n = hi - lo
                shape = np.exp(-3.0 * np.maximum(np.arange(n) - (fr - lo) - 3, 0) / max(n, 1))
            H[k, lo:hi] = np.maximum(H[k, lo:hi], shape)
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
    return W, H, windows


def update(V: np.ndarray, W: np.ndarray, H: np.ndarray, cfg: NMFDConfig, H_mask: np.ndarray | None = None,
           n_drums: int | None = None, W_mask: np.ndarray | None = None):
    """KL ダイバージェンスの乗法更新を 1 回。

    `H_mask` があれば活性に掛ける（"support" では窓の中 1・外 off_onset。"ramp" では距離で下がる係数）。
    `n_drums` はドラムの成分の数（残りは残差。残差の活性には `cfg.residual_penalty` の罰を掛ける）。
    `W_mask`（[K, F, 1] の 0/1）があれば、ひな形の持てる帯域をそれで限る。
    """
    K, F, L = W.shape
    T = H.shape[1]
    eps = 1e-10
    W2 = W.transpose(1, 0, 2).reshape(F, K * L)  # [F, K·L]
    lam = W2 @ _shifted(H, L).reshape(K * L, T) + eps
    R = V / lam
    # H の更新: numH[k, t] = Σ_τ Σ_f W[k, f, τ] R[f, t + τ]
    G = (W2.T @ R).reshape(K, L, T)
    colsum = W.sum(axis=1)  # [K, L]
    numH = np.zeros_like(H)
    denH = np.zeros_like(H)
    for tau in range(L):
        numH[:, : T - tau] += G[:, tau, tau:]
        denH[:, : T - tau] += colsum[:, tau : tau + 1]
    if cfg.residual and cfg.residual_penalty > 0 and n_drums is not None:
        denH[n_drums:] *= 1.0 + cfg.residual_penalty
    H = H * (numH / (denH + eps))
    if H_mask is not None:
        H = H * H_mask
    Hs = _shifted(H, L).reshape(K * L, T)
    lam = W2 @ Hs + eps
    R = V / lam
    # W の更新: numW[k, f, τ] = Σ_t R[f, t + τ] H[k, t]
    numW = (R @ Hs.T).reshape(F, K, L).transpose(1, 0, 2)  # [K, F, L]
    denW = Hs.sum(axis=1).reshape(K, 1, L) + eps
    W = W * (numW / denW)
    W = np.maximum(W, 1e-12)
    if W_mask is not None:
        W = W * W_mask
    W /= W.sum(axis=(1, 2), keepdims=True) + 1e-12  # 大きさは H に持たせる
    return W, H


def _factorize(x: np.ndarray, sr: int, hits: list[Hit], cfg: NMFDConfig, parts: tuple[str, ...], constrain_onsets: bool):
    """因子分解まで。返り値: (複素 STFT X, ひな形 W, 活性 H, ドラムの成分の数)。"""
    freqs, _, X = _stft(x, cfg)
    freqs = freqs * sr  # scipy は fs=1 のとき正規化周波数を返す
    V = np.abs(X) ** cfg.power + 1e-8
    frame_rate = sr / cfg.hop
    W, H, windows = initialize(V, hits, frame_rate, cfg, parts, freqs)
    n_drums = len(parts)
    H_mask = None
    if constrain_onsets:
        H_mask = np.ones_like(H)
        if cfg.constraint == "ramp":
            # 初期の活性（打点からの距離で 1 → off_onset）をそのままマスクにする（毎回掛かるので累積する。従来）
            scale = H[:n_drums].max(axis=1, keepdims=True) + 1e-12
            H_mask[:n_drums] = np.clip(H[:n_drums] / scale, cfg.off_onset, 1.0)
        else:
            # 窓の中は 1（自由）、外は off_onset。0/1 に近いので、何度掛けても窓の中の活性は減らない
            H_mask[:n_drums] = cfg.off_onset
            for k, wins in enumerate(windows):
                for lo, _, hi in wins:
                    H_mask[k, lo:hi] = 1.0
    W_mask = None
    low_cut = dict(cfg.template_low_cut_hz)
    if any(c in low_cut for c in parts):
        W_mask = np.ones((W.shape[0], W.shape[1], 1))
        for k, cls in enumerate(parts):
            if cls in low_cut:
                W_mask[k, freqs < low_cut[cls], 0] = 0.0
        W = W * W_mask
        W /= W.sum(axis=(1, 2), keepdims=True) + 1e-12
    for _ in range(cfg.iterations):
        W, H = update(V, W, H, cfg, H_mask, n_drums, W_mask)
    return X, W, H, n_drums


def _component_powers(W: np.ndarray, H: np.ndarray, cfg: NMFDConfig):
    """成分ごとの近似（Wiener マスクの分子）を 1 つずつ返す。全部を同時には持たない。"""
    for k in range(W.shape[0]):
        yield _approx(W[k : k + 1], H[k : k + 1]) ** cfg.mask_exponent


def compute_masks(x: np.ndarray, sr: int, hits: list[Hit], cfg: NMFDConfig | None = None, parts: tuple[str, ...] = PARTS,
                  constrain_onsets: bool = True) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """因子分解して、パートごとの時間周波数マスクを返す: (複素 STFT X [F, T], マスク [K, F, T]（和は 1）, 名前)。

    マスクは同じ設定の STFT に掛ければ別の信号にも使える（評価で、ドラムとにじみがそれぞれどのパートへ行くかを測る）。
    `constrain_onsets` が False なら活性を縛らない（打点は初期値にだけ使う）。マスクを全部並べるので、長い曲では `separate` を使う。
    """
    cfg = cfg or NMFDConfig()
    x = np.asarray(x, dtype=np.float64).ravel()
    X, W, H, _ = _factorize(x, sr, hits, cfg, parts, constrain_onsets)
    comps = np.stack(list(_component_powers(W, H, cfg)))
    total = comps.sum(axis=0)
    tiny = total < 1e-20
    masks = comps / np.where(tiny, 1.0, total)[None]
    masks[:, tiny] = 1.0 / comps.shape[0]  # どの成分も無いところは等分（マスクの和を常に 1 にし、パートの和を元に等しくする）
    names = list(parts) + (["residual"] if cfg.residual else [])
    return X, masks, names


def apply_masks(x: np.ndarray, masks: np.ndarray, names: list[str], cfg: NMFDConfig) -> dict[str, np.ndarray]:
    """`compute_masks` のマスクを信号 `x`（同じ長さ）に掛けて、名前 → 波形 を返す。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    _, _, X = _stft(x, cfg)
    return {name: _istft(X * m, cfg, x.size) for name, m in zip(names, masks)}


def separate(x: np.ndarray, sr: int, hits: list[Hit], cfg: NMFDConfig | None = None, parts: tuple[str, ...] = PARTS,
             constrain_onsets: bool = True) -> dict[str, np.ndarray]:
    """ドラムのステムをパートに分ける。返り値: 種類 → 波形（元と同じ長さ）。`residual` があれば "residual" も返す。"""
    cfg = cfg or NMFDConfig()
    x = np.asarray(x, dtype=np.float64).ravel()
    X, W, H, _ = _factorize(x, sr, hits, cfg, parts, constrain_onsets)
    # Wiener マスクをパートごとに順に作る（[K, F, T] を一度に持たない。3 分の曲でもメモリは STFT 数枚ぶん）
    total = np.zeros(X.shape)
    for comp in _component_powers(W, H, cfg):
        total += comp
    tiny = total < 1e-20
    total[tiny] = 1.0
    names = list(parts) + (["residual"] if cfg.residual else [])
    out: dict[str, np.ndarray] = {}
    for name, comp in zip(names, _component_powers(W, H, cfg)):
        mask = comp / total
        mask[tiny] = 1.0 / len(names)  # どの成分も無いところは等分（マスクの和を常に 1 にし、パートと残差の和を元に等しくする）
        out[name] = _istft(X * mask, cfg, x.size)
    return out


def snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    """SNR（dB）。ref と est は同じ長さ。"""
    n = min(ref.size, est.size)
    r, e = ref[:n], est[:n]
    return float(10.0 * np.log10((np.sum(r**2) + 1e-12) / (np.sum((r - e) ** 2) + 1e-12)))
