"""ONNX の採譜モデルを ONNX Runtime で動かして打点を出す（プラグイン側と同じ経路）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import FeatureConfig, log_mel, pick_hits
from .labels import Hit


def load_meta(onnx_path: str | Path) -> tuple[FeatureConfig, tuple[str, ...], float]:
    meta = json.loads(Path(onnx_path).with_suffix(".json").read_text(encoding="utf-8"))
    cfg = FeatureConfig(**meta["features"])
    return cfg, tuple(meta["classes"]), float(meta.get("threshold", 0.5))


def transcribe_onnx(x: np.ndarray, sr: int, onnx_path: str | Path, threshold: float | None = None) -> list[Hit]:
    import onnxruntime as ort

    cfg, classes, thr = load_meta(onnx_path)
    if sr != cfg.sr:
        n_new = int(round(x.size * cfg.sr / sr))
        x = np.interp(np.linspace(0, x.size - 1, n_new), np.arange(x.size), x)
    feat = log_mel(x, cfg)[None]  # [1, T, M]
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    prob = sess.run(None, {sess.get_inputs()[0].name: feat})[0][0]
    return pick_hits(prob, cfg, classes, thr if threshold is None else threshold)
