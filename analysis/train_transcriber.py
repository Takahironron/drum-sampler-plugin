#!/usr/bin/env python3
"""合成データで小さな CRNN を学習し、ONNX に書き出す（計画書 3.2 の道具立ての確認）。

使い方::

    python train_transcriber.py [--steps 600] [--out models/crnn_synth.onnx] [--seed 0]

- 学習データは `drumeval.synth` で乱数のパターンを描画したもの（音色の揺らぎ・にじみ・残響つき）。
  実曲の精度を主張するものではなく、「学習 → ONNX → ONNX Runtime で同じ結果 → F 値」が通ることの確認。
- E-GMD などの実データで学習するときは `make_batch` を差し替える。
- 出力: `<out>.onnx` と、特徴量の設定・種類・しきい値を書いた `<out>.json`。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drumeval import Hit, m0_transcription_score, transcription_scores  # noqa: E402
from drumeval.features import FeatureConfig, frame_targets, log_mel, pick_hits  # noqa: E402
from drumeval.labels import M0_CLASSES  # noqa: E402
from drumeval.synth import render  # noqa: E402

CLASSES = M0_CLASSES  # kick, snare, hihat


def random_pattern(rng: np.random.Generator, seconds: float = 6.0) -> list[Hit]:
    """16分のグリッドに、種類ごとの確率で打を置く。テンポ・強さ・ゴースト・休符を乱数で。"""
    bpm = rng.uniform(75, 170)
    step = 60.0 / bpm / 4
    n_steps = int(seconds / step)
    p_kick = rng.uniform(0.15, 0.5)
    p_snare = rng.uniform(0.1, 0.35)
    p_hat = rng.uniform(0.3, 0.9)
    hits: list[Hit] = []
    t0 = rng.uniform(0.05, 0.3)
    for i in range(n_steps):
        t = t0 + i * step + rng.normal(0, 0.004)  # 揺れ
        if t >= seconds - 0.2:
            break
        beat_pos = i % 4
        if rng.random() < (p_kick * (2.0 if beat_pos == 0 else 0.6)):
            hits.append(Hit(t, "kick", int(rng.integers(50, 128))))
        if rng.random() < (p_snare * (2.5 if (i % 8) == 4 else 0.4)):
            hits.append(Hit(t, "snare", int(rng.integers(40, 128))))
        if rng.random() < (p_hat * (1.0 if beat_pos % 2 == 0 else 0.6)):
            hits.append(Hit(t, "hihat_closed", int(rng.integers(30, 120))))
    return hits


def make_example(rng: np.random.Generator, cfg: FeatureConfig, seconds: float = 6.0) -> tuple[np.ndarray, np.ndarray, list[Hit]]:
    hits = random_pattern(rng, seconds)
    x = render(hits, sr=cfg.sr, duration=seconds, bleed=float(rng.uniform(0, 0.5)),
               reverb_tail=float(rng.choice([0.0, rng.uniform(0.1, 0.6)])), seed=int(rng.integers(1 << 30)),
               variation=float(rng.uniform(0.3, 1.0)))
    x = x * 10 ** (rng.uniform(-12, 0) / 20)
    feat = log_mel(x, cfg)
    y = frame_targets(hits, feat.shape[0], cfg, CLASSES)
    return feat, y, hits


def make_batch(rng: np.random.Generator, cfg: FeatureConfig, batch: int, seconds: float):
    feats, ys = [], []
    for _ in range(batch):
        f, y, _ = make_example(rng, cfg, seconds)
        feats.append(f)
        ys.append(y)
    t = min(f.shape[0] for f in feats)
    return np.stack([f[:t] for f in feats]), np.stack([y[:t] for y in ys])


def evaluate(model, cfg: FeatureConfig, rng: np.random.Generator, n: int = 8, threshold: float = 0.5) -> dict:
    import torch

    model.eval()
    scores = []
    with torch.no_grad():
        for _ in range(n):
            feat, _, hits = make_example(rng, cfg, 8.0)
            prob = model(torch.from_numpy(feat[None]))[0].numpy()
            est = pick_hits(prob, cfg, CLASSES, threshold)
            s = transcription_scores(hits, est, CLASSES)
            scores.append([s[c].f_measure for c in CLASSES])
    model.train()
    m = np.mean(scores, axis=0)
    return {c: float(v) for c, v in zip(CLASSES, m)} | {"m0_1": float(m.mean())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "models" / "crnn_synth.onnx"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=100)
    args = ap.parse_args(argv)

    import torch
    import torch.nn.functional as F

    from drumeval.model import DrumCRNN

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cfg = FeatureConfig()
    model = DrumCRNN(cfg.n_mels, len(CLASSES))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    pos_weight = torch.tensor([8.0] * len(CLASSES))  # 打点のフレームは少ないので重みを付ける

    t_start = time.time()
    for step in range(1, args.steps + 1):
        feat, y = make_batch(rng, cfg, args.batch, args.seconds)
        xb, yb = torch.from_numpy(feat), torch.from_numpy(y)
        prob = model(xb)
        loss = F.binary_cross_entropy(prob, yb, weight=1.0 + (pos_weight - 1.0) * yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(model, cfg, np.random.default_rng(1000 + step))
            print(f"step {step:5d}  loss {loss.item():.4f}  F kick {ev['kick']:.3f} snare {ev['snare']:.3f} hihat {ev['hihat']:.3f}  M0-1 {ev['m0_1']:.3f}  ({time.time() - t_start:.0f}s)", flush=True)

    # ONNX に書き出し、ONNX Runtime で同じ結果になることを確かめる
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.zeros(1, 200, cfg.n_mels)
    torch.onnx.export(model, dummy, str(out), input_names=["log_mel"], output_names=["prob"],
                      dynamic_axes={"log_mel": {0: "batch", 1: "frames"}, "prob": {0: "batch", 1: "frames"}},
                      opset_version=17, dynamo=False)
    meta = {"features": cfg.__dict__, "classes": list(CLASSES), "threshold": 0.5,
            "trained_on": "synthetic (drumeval.synth)", "steps": args.steps, "seed": args.seed}
    out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    import onnxruntime as ort

    feat, _, hits = make_example(np.random.default_rng(999), cfg, 8.0)
    with torch.no_grad():
        p_torch = model(torch.from_numpy(feat[None]))[0].numpy()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    p_ort = sess.run(None, {"log_mel": feat[None]})[0][0]
    diff = float(np.max(np.abs(p_torch - p_ort)))
    f_ort = m0_transcription_score(hits, pick_hits(p_ort, cfg, CLASSES))
    print(f"ONNX 書き出し: {out}（{out.stat().st_size / 1024:.0f} KB）  torch と ORT の最大差 {diff:.2e}  ORT の M0-1 {f_ort:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
