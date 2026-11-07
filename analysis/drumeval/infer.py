"""ONNX の採譜モデルを ONNX Runtime で動かして打点を出す（プラグイン側と同じ経路）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import FeatureConfig, log_mel, pick_hits
from .labels import Hit


def resample_to(x: np.ndarray, sr: int, sr_target: int) -> np.ndarray:
    """学習（`egmd.load_audio`）と同じ多相フィルタで間引く。線形補間（np.interp）だと高域が残って学習時と入力がずれる。
    C++ に移すときも同じ（窓付き sinc か多相）にする。"""
    from math import gcd

    from scipy.signal import resample_poly

    if sr == sr_target:
        return x
    g = gcd(int(sr), int(sr_target))
    return resample_poly(np.asarray(x, dtype=np.float64), sr_target // g, sr // g)


def load_meta(onnx_path: str | Path) -> tuple[FeatureConfig, tuple[str, ...], float | dict[str, float]]:
    """返り値のしきい値は、json に `thresholds`（種類別）があればその辞書、無ければ `threshold`（1 つ）。"""
    meta = json.loads(Path(onnx_path).with_suffix(".json").read_text(encoding="utf-8"))
    cfg = FeatureConfig(**meta["features"])
    thr = meta.get("thresholds") or float(meta.get("threshold", 0.5))
    return cfg, tuple(meta["classes"]), thr


def transcribe_onnx(x: np.ndarray, sr: int, onnx_path: str | Path, threshold: float | None = None) -> list[Hit]:
    import onnxruntime as ort

    cfg, classes, thr = load_meta(onnx_path)
    if sr != cfg.sr:
        x = resample_to(x, sr, cfg.sr)
    feat = log_mel(x, cfg)[None]  # [1, T, M]
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    prob = sess.run(None, {sess.get_inputs()[0].name: feat})[0][0]
    return pick_hits(prob, cfg, classes, thr if threshold is None else threshold)
