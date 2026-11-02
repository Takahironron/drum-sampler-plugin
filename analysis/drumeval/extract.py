"""一打の切り出しと強さ別の層（計画書 3.4 の候補 (a) の最小版）。

手順:
1. 採譜の結果（打点ラベル）から「前後に他の打が重ならない打」を選ぶ（`select_isolated_hits`）。
   前 `pre_sec`（既定 0.1 秒。直前の打の尾が混ざらないように）以内と後 `post_sec`（既定 0.15 秒）以内に、種類を問わず他の打が無いもの。
2. 打点の直前から次の打の直前まで（上限 `max_len_sec`）を切り出し、短いフェードを付ける（`cut_segments`）。
3. 種類ごとにピークの大きさで並べ、強さ別の層に分ける。同じ層の複数の打はラウンドロビンにする（`build_kit`）。
4. 切り出したキットで打点ラベルを鳴らし直す（`render_kit`）。元のステムとの STFT 距離が「信頼度」の材料になる。

音の尾は次の打で切れるので、延長（モーダル合成か減衰の当てはめ）は Phase 2 の課題。ここではやらない。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .labels import COARSE_CLASSES, Hit, to_coarse


@dataclass
class Segment:
    cls: str  # 粗い種類
    time: float  # 元の打点（秒）
    audio: np.ndarray  # モノラル
    peak_db: float


@dataclass
class Layer:
    vel_lo: int
    vel_hi: int
    samples: list[np.ndarray] = field(default_factory=list)  # ラウンドロビン
    source_times: list[float] = field(default_factory=list)
    peaks_db: list[float] = field(default_factory=list)  # 各音のピーク（音量合わせ用）
    features: list[list[float]] = field(default_factory=list)  # 各音の特徴（近い音を選ぶ用。無ければ空）


@dataclass
class Kit:
    sr: int
    parts: dict[str, list[Layer]]  # 種類 → 弱い層から強い層へ
    feat_norm: dict[str, tuple[list[float], list[float]]] = field(default_factory=dict)  # 種類 → (平均, 標準偏差)。特徴の正規化

    def n_samples(self, cls: str) -> int:
        return sum(len(l.samples) for l in self.parts.get(cls, []))

    def all_samples(self, cls: str) -> list[tuple[np.ndarray, float, list[float] | None]]:
        out = []
        for l in self.parts.get(cls, []):
            for j, smp in enumerate(l.samples):
                peak = l.peaks_db[j] if j < len(l.peaks_db) else 20.0 * np.log10(np.max(np.abs(smp)) + 1e-9)
                feat = l.features[j] if j < len(l.features) else None
                out.append((smp, peak, feat))
        return out

    def pick_nearest(self, cls: str, feature: np.ndarray, exclude_time: float | None = None) -> tuple[np.ndarray, float] | None:
        """特徴の近い音を返す（音, ピーク dB）。特徴の無いキットなら None。

        `exclude_time` を渡すと、その時刻（±5 ms）に由来する音を候補から外す（自分自身を除く上限の測定用）。
        """
        cands = []
        for l in self.parts.get(cls, []):
            for j, smp in enumerate(l.samples):
                if j >= len(l.features):
                    continue
                if exclude_time is not None and j < len(l.source_times) and abs(l.source_times[j] - exclude_time) < 0.005:
                    continue
                peak = l.peaks_db[j] if j < len(l.peaks_db) else 20.0 * np.log10(np.max(np.abs(smp)) + 1e-9)
                cands.append((smp, peak, l.features[j]))
        if not cands or cls not in self.feat_norm:
            return None
        mean, std = (np.asarray(v, dtype=float) for v in self.feat_norm[cls])
        q = (np.asarray(feature, dtype=float) - mean) / std
        d = [np.sum(((np.asarray(f) - mean) / std - q) ** 2) for _, _, f in cands]
        smp, peak, _ = cands[int(np.argmin(d))]
        return smp, peak


CHOKE_CLASSES = {"kick", "snare", "hihat", "tom"}  # 同じ種類の次の打で音を止める（シンバルは重ねる）


def select_isolated_hits(hits: list[Hit], pre_sec: float = 0.1, post_sec: float = 0.15) -> list[tuple[Hit, float]]:
    """重ならない打と、その打に使える長さ（次の打までの秒数）を返す。"""
    hs = sorted(hits, key=lambda h: h.time)
    out: list[tuple[Hit, float]] = []
    for i, h in enumerate(hs):
        prev_t = hs[i - 1].time if i > 0 else -np.inf
        next_t = hs[i + 1].time if i + 1 < len(hs) else np.inf
        if h.time - prev_t < pre_sec or next_t - h.time < post_sec:
            continue
        out.append((h, float(next_t - h.time)))
    return out


def cut_segments(
    x: np.ndarray,
    sr: int,
    hits: list[Hit],
    pre_sec: float = 0.1,
    post_sec: float = 0.15,
    max_len_sec: float = 1.5,
    lead_sec: float = 0.002,
    fade_out_sec: float = 0.01,
) -> list[Segment]:
    x = np.asarray(x, dtype=np.float64).ravel()
    segs: list[Segment] = []
    for h, avail in select_isolated_hits(hits, pre_sec, post_sec):
        start = max(int(round((h.time - lead_sec) * sr)), 0)
        length = int(min(avail - lead_sec, max_len_sec) * sr)
        end = min(start + length, x.size)
        if end - start < int(0.01 * sr):
            continue
        a = x[start:end].copy()
        n_in = max(int(0.001 * sr), 1)
        a[:n_in] *= np.linspace(0.0, 1.0, n_in)
        n_out = min(int(fade_out_sec * sr), a.size)
        a[-n_out:] *= np.linspace(1.0, 0.0, n_out)
        peak = float(np.max(np.abs(a)) + 1e-9)
        segs.append(Segment(to_coarse(h.cls), h.time, a, 20.0 * np.log10(peak)))
    return segs


def cut_segments_from_parts(
    parts: dict[str, tuple[np.ndarray, int]],
    hits: list[Hit],
    pre_sec: float = 0.1,
    post_sec: float = 0.15,
    max_len_sec: float = 1.5,
) -> list[Segment]:
    """パート別に分離された音（種類 → (波形, sr)）から切り出す。

    他の種類の打は別のステムにあるので、重なりの判定は同じ種類の打どうしだけで行う。
    混ざったドラムから切るより、切り出せる打がずっと多くなる。
    """
    segs: list[Segment] = []
    for cls, (x, sr) in parts.items():
        own = [h for h in hits if to_coarse(h.cls) == cls]
        segs += cut_segments(x, sr, own, pre_sec=pre_sec, post_sec=post_sec, max_len_sec=max_len_sec)
    return segs


def self_render_from_parts(
    parts: dict[str, tuple[np.ndarray, int]],
    hits: list[Hit],
    duration: float,
    max_len_sec: float = 1.5,
    fade_out_sec: float = 0.01,
) -> np.ndarray:
    """各打を「その打自身の区間」（打点から同じ種類の次の打まで）で鳴らし直す（自己再現）。

    層やラウンドロビンの量子化が無いので、元との距離は「打点の誤り・取りこぼし」と「区間の切り方」だけを映す。
    キットでの再現との差が、層とラウンドロビンで失う分。
    """
    sr = next(iter(parts.values()))[1]
    n = int(duration * sr)
    out = np.zeros(n)
    for cls, (x, psr) in parts.items():
        own = sorted((h for h in hits if to_coarse(h.cls) == cls), key=lambda h: h.time)
        for i, h in enumerate(own):
            start = int(round(h.time * psr))
            next_t = own[i + 1].time if i + 1 < len(own) else h.time + max_len_sec
            end = min(start + int(min(next_t - h.time, max_len_sec) * psr), x.size, n)
            if end - start < int(0.005 * psr):
                continue
            a = x[start:end].copy()
            n_out = min(int(fade_out_sec * psr), a.size)
            a[-n_out:] *= np.linspace(1.0, 0.0, n_out)
            out[start:end] += a
    return out


FEATURE_NAMES = ("peak_db", "log_decay", "log_centroid", "low_db", "mid_db", "high_db")


def segment_features(a: np.ndarray, sr: int) -> np.ndarray:
    """一打の特徴: [ピーク dB, log10(減衰時間 秒), log(スペクトル重心 Hz), 低域 dB, 中域 dB, 高域 dB]。

    減衰時間は 5 ms 窓の RMS が最大から 20 dB 下がるまでの時間。重心は最初の 30 ms のスペクトル。
    帯域のエネルギー（低 <150 Hz、中 150〜2 kHz、高 >5 kHz）は最初の 60 ms の RMS（dB）で、キックの胴鳴りの量や
    スネアの響き線の量の違いを見る。
    """
    a = np.asarray(a, dtype=np.float64)
    peak = float(np.max(np.abs(a)) + 1e-9)
    win = max(int(0.005 * sr), 1)
    n = a.size // win
    if n >= 1:
        env = np.sqrt(np.mean(a[: n * win].reshape(n, win) ** 2, axis=1) + 1e-12)
        env_db = 20.0 * np.log10(env)
        i_max = int(np.argmax(env_db))
        below = np.flatnonzero(env_db[i_max:] < env_db[i_max] - 20.0)
        decay = (i_max + below[0]) * win / sr if below.size else a.size / sr
    else:
        decay = a.size / sr
    head = a[: max(int(0.03 * sr), 64)]
    spec = np.abs(np.fft.rfft(head * np.hanning(head.size)))
    f = np.fft.rfftfreq(head.size, 1.0 / sr)
    centroid = float(np.sum(f * spec) / (np.sum(spec) + 1e-12)) + 1.0
    body = a[: max(int(0.06 * sr), 128)]
    spec_b = np.abs(np.fft.rfft(body * np.hanning(body.size))) ** 2
    f_b = np.fft.rfftfreq(body.size, 1.0 / sr)
    bands = []
    for lo, hi in ((20.0, 150.0), (150.0, 2000.0), (5000.0, sr / 2)):
        sel = (f_b >= lo) & (f_b < hi)
        bands.append(10.0 * np.log10(np.mean(spec_b[sel]) + 1e-12) if np.any(sel) else -120.0)
    return np.array([20.0 * np.log10(peak), np.log10(decay + 1e-3), np.log(centroid), *bands])


def head_embedding(a: np.ndarray, sr: int, head_sec: float = 0.25, n_mels: int = 40, n_fft: int = 1024, hop: int = 512) -> np.ndarray:
    """一打の先頭 `head_sec` 秒の対数メルスペクトログラムを 1 本のベクトルにした埋め込み。

    6 個の特徴（`segment_features`）より細かく音色を表す。計画書 3.4 の候補 (b)「埋め込みで近い音を探す」の最小版。
    長さは固定（短い音は無音で埋める）。音量の影響を減らすため、全体の最大値を 0 dB に合わせる。
    """
    from .features import FeatureConfig, mel_filterbank

    n = int(head_sec * sr)
    x = np.asarray(a, dtype=np.float64)[:n]
    x = np.pad(x, (0, max(n - x.size, 0)))
    x = x / (np.max(np.abs(x)) + 1e-9)
    cfg = FeatureConfig(sr=sr, n_fft=n_fft, hop=hop, n_mels=n_mels, f_min=30.0, f_max=min(16000.0, sr / 2 * 0.95))
    from scipy.signal import stft

    _, _, z = stft(x, nperseg=n_fft, noverlap=n_fft - hop, window="hann", padded=True, boundary="zeros")
    mel = mel_filterbank(cfg) @ (np.abs(z) ** 2)
    return np.log1p(1000.0 * mel).ravel()


def _kmeans(x: np.ndarray, k: int, rng: np.random.Generator, iters: int = 30) -> np.ndarray:
    """小さな k-means（k-means++ の初期化）。返り値は各中心に最も近い点の添字（代表）。"""
    n = x.shape[0]
    k = min(k, n)
    centers = [x[int(rng.integers(n))]]
    for _ in range(1, k):
        d2 = np.min([np.sum((x - c) ** 2, axis=1) for c in centers], axis=0)
        probs = d2 / (d2.sum() + 1e-12) if d2.sum() > 0 else np.full(n, 1.0 / n)
        centers.append(x[int(rng.choice(n, p=probs))])
    centers = np.asarray(centers)
    for _ in range(iters):
        assign = np.argmin([np.sum((x - c) ** 2, axis=1) for c in centers], axis=0)
        new = np.array([x[assign == j].mean(axis=0) if np.any(assign == j) else centers[j] for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    reps = sorted({int(np.argmin(np.sum((x - c) ** 2, axis=1))) for c in centers})
    return np.asarray(reps)


def build_kit_by_features(segments: list[Segment], sr: int, n_per_class: int | dict[str, int] = 12, seed: int = 0,
                          classes: tuple[str, ...] = COARSE_CLASSES, feature_fn=None) -> Kit:
    """種類ごとに、特徴空間の k-means の代表 `n_per_class` 音でキットを作る。

    `feature_fn(audio, sr)` は既定で `segment_features`（音量・減衰・明るさ・帯域の 6 個）。`head_embedding` を渡すと
    先頭 0.25 秒の対数メルで選ぶ。3 層 × 3 音のように音量だけで選ぶのではなく、奏法の違いも代表に残る。
    再生時は各打の特徴に最も近い音を選び、音量を合わせる（`render_kit(..., hit_features=...)`。同じ feature_fn を使うこと）。
    """
    feature_fn = feature_fn or segment_features
    rng = np.random.default_rng(seed)
    parts: dict[str, list[Layer]] = {}
    feat_norm: dict[str, tuple[list[float], list[float]]] = {}
    for cls in classes:
        segs = [s for s in segments if s.cls == cls]
        if not segs:
            continue
        feats = np.array([feature_fn(s.audio, sr) for s in segs])
        mean = feats.mean(axis=0)
        std = feats.std(axis=0) + 1e-3
        z = (feats - mean) / std
        n_cls = n_per_class.get(cls, 12) if isinstance(n_per_class, dict) else n_per_class
        reps = _kmeans(z, n_cls, rng) if 0 < n_cls < len(segs) else np.arange(len(segs))  # n_cls <= 0 なら全部を代表にする
        layer = Layer(1, 127)
        for i in sorted(reps, key=lambda i: feats[i, 0]):
            layer.samples.append(segs[i].audio)
            layer.source_times.append(segs[i].time)
            layer.peaks_db.append(float(feats[i, 0]))
            layer.features.append([float(v) for v in feats[i]])
        parts[cls] = [layer]
        feat_norm[cls] = ([float(v) for v in mean], [float(v) for v in std])
    return Kit(sr, parts, feat_norm)


def hit_segments_from_parts(parts: dict[str, tuple[np.ndarray, int]], hits: list[Hit], max_len_sec: float = 1.5) -> list[np.ndarray | None]:
    """各打の「自分の区間」（打点から同じ種類の次の打まで、上限 max_len_sec）をパート別ステムから取る。hits と同じ順。"""
    out: list[np.ndarray | None] = [None] * len(hits)
    by_cls: dict[str, list[int]] = {}
    for i, h in enumerate(hits):
        by_cls.setdefault(to_coarse(h.cls), []).append(i)
    for cls, idx in by_cls.items():
        if cls not in parts:
            continue
        x, sr = parts[cls]
        idx = sorted(idx, key=lambda i: hits[i].time)
        for k, i in enumerate(idx):
            t = hits[i].time
            next_t = hits[idx[k + 1]].time if k + 1 < len(idx) else t + max_len_sec
            start = int(round(t * sr))
            end = min(start + int(min(next_t - t, max_len_sec) * sr), x.size)
            if end - start >= int(0.005 * sr):
                out[i] = x[start:end]
    return out


def build_kit(
    segments: list[Segment],
    sr: int,
    n_layers: int = 3,
    round_robin: int = 3,
    classes: tuple[str, ...] = COARSE_CLASSES,
) -> Kit:
    """種類ごとにピークの大きさで層に分ける。打が少ない種類は層を減らす。"""
    parts: dict[str, list[Layer]] = {}
    for cls in classes:
        segs = sorted((s for s in segments if s.cls == cls), key=lambda s: s.peak_db)
        if not segs:
            continue
        k = max(1, min(n_layers, len(segs)))
        # ピークの分位で層の境界を決める
        bounds = np.linspace(0, len(segs), k + 1).astype(int)
        layers: list[Layer] = []
        for li in range(k):
            chunk = segs[bounds[li] : bounds[li + 1]]
            if not chunk:
                continue
            vel_lo = 1 + int(round(126 * li / k))
            vel_hi = int(round(127 * (li + 1) / k))
            # 層の中では代表的な（中央寄りの）打からラウンドロビンに使う
            mid = len(chunk) // 2
            order = sorted(range(len(chunk)), key=lambda j: abs(j - mid))[:round_robin]
            layer = Layer(vel_lo, vel_hi)
            for j in sorted(order):
                layer.samples.append(chunk[j].audio)
                layer.source_times.append(chunk[j].time)
                layer.peaks_db.append(chunk[j].peak_db)
            layers.append(layer)
        parts[cls] = layers
    return Kit(sr, parts)


def _onset_lag(target: np.ndarray, sample: np.ndarray, sr: int, max_lag_sec: float = 0.015) -> int:
    """`sample` を何サンプルずらせば立ち上がりが `target` に重なるか（相互相関、±max_lag_sec）。正なら遅らせる。"""
    n = min(int(0.04 * sr), target.size, sample.size)
    if n < 16:
        return 0
    a = np.abs(target[:n]).astype(np.float64)
    b = np.abs(sample[:n]).astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    max_lag = min(int(max_lag_sec * sr), n // 2)  # 短い音では、ずらす幅を比較する長さの半分までに抑える
    if max_lag <= 0:
        return 0
    best, best_lag = -np.inf, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            c = float(np.dot(a[lag:], b[: n - lag]))
        else:
            c = float(np.dot(a[: n + lag], b[-lag:]))
        if c > best:
            best, best_lag = c, lag
    return best_lag


def render_kit(
    hits: list[Hit],
    kit: Kit,
    duration: float | None = None,
    hit_levels_db: list[float | None] | None = None,
    hit_features: list[np.ndarray | None] | None = None,
    gain_match: bool = False,
    choke: bool | set[str] = False,
    max_gain_db: float = 12.0,
    hit_audio: list[np.ndarray | None] | None = None,
    align: bool = False,
    leave_one_out: bool = False,
) -> np.ndarray:
    """打点ラベルをキットで鳴らす。

    - 音の選び方: `hit_features` があれば（特徴の入ったキットで）最も近い音。無ければ強さの層とラウンドロビン。
      強さが無い打は最も強い層を使う。
    - `gain_match`: `hit_levels_db`（その打の元のピーク dB）に合わせて音量を調整する（±max_gain_db まで）。
    - `choke`: 同じ種類の次の打が来たら前の音を 5 ms で止める。True なら `CHOKE_CLASSES`（シンバル以外）、
      種類の集合（例: {"hihat"}）ならその種類だけ、False なら止めない。
    - `align`: `hit_audio`（その打自身の区間）があれば、選んだ音の立ち上がりを相互相関で元に合わせて置く（±15 ms）。
      打点の推定は約 10 ms 刻みなので、そのままだと最大 1 フレームずれる。
    - `leave_one_out`: 選ぶときに、その打自身に由来する音を候補から外す（全部を代表にしたときの上限の測定用）。
    `hit_levels_db`／`hit_features`／`hit_audio` は `hits` と同じ順の並び。
    """
    sr = kit.sr
    last = max((h.time for h in hits), default=0.0)
    if duration is None:
        duration = last + 1.5
    n = int(duration * sr)
    out = np.zeros(n)
    rr_index: dict[tuple[str, int], int] = {}
    order = sorted(range(len(hits)), key=lambda i: hits[i].time)
    # 同じ種類の次の打の時刻（チョーク用）
    next_same: dict[int, float] = {}
    last_of_cls: dict[str, int] = {}
    for i in order:
        c = to_coarse(hits[i].cls)
        if c in last_of_cls:
            next_same[last_of_cls[c]] = hits[i].time
        last_of_cls[c] = i
    fade = max(int(0.005 * sr), 1)
    choke_set: set[str] = CHOKE_CLASSES if choke is True else (choke if isinstance(choke, set) else set())

    for i in order:
        h = hits[i]
        cls = to_coarse(h.cls)
        layers = kit.parts.get(cls)
        if not layers:
            continue
        chosen = None
        if hit_features is not None and hit_features[i] is not None:
            chosen = kit.pick_nearest(cls, hit_features[i], exclude_time=h.time if leave_one_out else None)
        if chosen is None:
            if h.velocity is None:
                li = len(layers) - 1
            else:
                li = next((k for k, l in enumerate(layers) if l.vel_lo <= h.velocity <= l.vel_hi), len(layers) - 1)
            layer = layers[li]
            key = (cls, li)
            j = rr_index.get(key, 0) % len(layer.samples)
            rr_index[key] = j + 1
            peak = layer.peaks_db[j] if j < len(layer.peaks_db) else 20.0 * np.log10(np.max(np.abs(layer.samples[j])) + 1e-9)
            chosen = (layer.samples[j], peak)
        s, peak_db = chosen
        g = 1.0
        if gain_match and hit_levels_db is not None and hit_levels_db[i] is not None:
            g = 10 ** (float(np.clip(hit_levels_db[i] - peak_db, -max_gain_db, max_gain_db)) / 20)
        start = int(round(h.time * sr))
        if align and hit_audio is not None and hit_audio[i] is not None:
            start += _onset_lag(hit_audio[i], s, sr)
        if start >= n or start < 0:
            continue
        length = min(s.size, n - start)
        if cls in choke_set and i in next_same:
            limit = int((next_same[i] - h.time) * sr) + fade
            length = min(length, max(limit, fade))
        seg = s[:length] * g
        if length < s.size:
            seg = seg.copy()
            k = min(fade, seg.size)
            seg[-k:] *= np.linspace(1.0, 0.0, k)
        out[start : start + length] += seg
    return out


def estimate_velocities(hits: list[Hit], segments_or_peaks: dict[str, list[tuple[float, float]]] | None, x: np.ndarray, sr: int) -> list[Hit]:
    """打点ごとの立ち上がりの大きさ（打点から 20 ms のピーク）を、種類ごとの分布で 1〜127 に割り当てる（計画書 3.2）。

    最小のピーク → 30、最大 → 127 の線形。打が1つしか無い種類は 100。
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    win = int(0.02 * sr)
    peaks = []
    for h in hits:
        s = int(round(h.time * sr))
        seg = x[max(s, 0) : s + win]
        peaks.append(20.0 * np.log10(np.max(np.abs(seg)) + 1e-9) if seg.size else -100.0)
    peaks_arr = np.asarray(peaks)
    out: list[Hit] = []
    for cls in {to_coarse(h.cls) for h in hits}:
        idx = [i for i, h in enumerate(hits) if to_coarse(h.cls) == cls]
        lo, hi = peaks_arr[idx].min(), peaks_arr[idx].max()
        for i in idx:
            if hi - lo < 1e-6:
                v = 100
            else:
                v = int(round(30 + (peaks_arr[i] - lo) / (hi - lo) * 97))
            out.append(Hit(hits[i].time, hits[i].cls, max(1, min(127, v))))
    out.sort(key=lambda h: h.time)
    return out


def save_kit(kit: Kit, out_dir: str | Path) -> None:
    import soundfile as sf

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta: dict = {"sr": kit.sr, "parts": {}}
    for cls, layers in kit.parts.items():
        (out / cls).mkdir(exist_ok=True)
        meta["parts"][cls] = []
        for li, layer in enumerate(layers):
            files = []
            for j, s in enumerate(layer.samples):
                name = f"{cls}/layer{li + 1}_rr{j + 1}.wav"
                sf.write(out / name, s.astype(np.float32), kit.sr)
                files.append(name)
            meta["parts"][cls].append({"vel_lo": layer.vel_lo, "vel_hi": layer.vel_hi, "files": files, "source_times": layer.source_times,
                                       "peaks_db": layer.peaks_db, "features": layer.features})
    meta["feat_norm"] = kit.feat_norm
    (out / "kit.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_kit(kit_dir: str | Path) -> Kit:
    import soundfile as sf

    d = Path(kit_dir)
    meta = json.loads((d / "kit.json").read_text(encoding="utf-8"))
    parts: dict[str, list[Layer]] = {}
    for cls, layers in meta["parts"].items():
        parts[cls] = []
        for l in layers:
            layer = Layer(l["vel_lo"], l["vel_hi"], source_times=list(l.get("source_times", [])),
                          peaks_db=list(l.get("peaks_db", [])), features=list(l.get("features", [])))
            for f in l["files"]:
                a, _ = sf.read(d / f, dtype="float64", always_2d=True)
                layer.samples.append(a.mean(axis=1))
            parts[cls].append(layer)
    feat_norm = {k: (list(v[0]), list(v[1])) for k, v in meta.get("feat_norm", {}).items()}
    return Kit(int(meta["sr"]), parts, feat_norm)
