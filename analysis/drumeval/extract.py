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


@dataclass
class Kit:
    sr: int
    parts: dict[str, list[Layer]]  # 種類 → 弱い層から強い層へ

    def n_samples(self, cls: str) -> int:
        return sum(len(l.samples) for l in self.parts.get(cls, []))


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
            layers.append(layer)
        parts[cls] = layers
    return Kit(sr, parts)


def render_kit(hits: list[Hit], kit: Kit, duration: float | None = None) -> np.ndarray:
    """打点ラベルをキットで鳴らす。強さが無い打は最も強い層を使う。"""
    sr = kit.sr
    last = max((h.time for h in hits), default=0.0)
    if duration is None:
        duration = last + 1.5
    n = int(duration * sr)
    out = np.zeros(n)
    rr_index: dict[tuple[str, int], int] = {}
    for h in sorted(hits, key=lambda h: h.time):
        cls = to_coarse(h.cls)
        layers = kit.parts.get(cls)
        if not layers:
            continue
        if h.velocity is None:
            li = len(layers) - 1
        else:
            li = next((i for i, l in enumerate(layers) if l.vel_lo <= h.velocity <= l.vel_hi), len(layers) - 1)
        layer = layers[li]
        key = (cls, li)
        j = rr_index.get(key, 0) % len(layer.samples)
        rr_index[key] = j + 1
        s = layer.samples[j]
        start = int(round(h.time * sr))
        end = min(start + s.size, n)
        if start < n:
            out[start:end] += s[: end - start]
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
            meta["parts"][cls].append({"vel_lo": layer.vel_lo, "vel_hi": layer.vel_hi, "files": files, "source_times": layer.source_times})
    (out / "kit.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_kit(kit_dir: str | Path) -> Kit:
    import soundfile as sf

    d = Path(kit_dir)
    meta = json.loads((d / "kit.json").read_text(encoding="utf-8"))
    parts: dict[str, list[Layer]] = {}
    for cls, layers in meta["parts"].items():
        parts[cls] = []
        for l in layers:
            layer = Layer(l["vel_lo"], l["vel_hi"], source_times=list(l.get("source_times", [])))
            for f in l["files"]:
                a, _ = sf.read(d / f, dtype="float64", always_2d=True)
                layer.samples.append(a.mean(axis=1))
            parts[cls].append(layer)
    return Kit(int(meta["sr"]), parts)
