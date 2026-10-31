#!/usr/bin/env python3
"""採譜モデル（小さな CRNN）を学習し、ONNX に書き出す。

2 つの学習データを選べる。

合成データ（道具立ての確認用。実曲の精度は出ない）::

    python train_transcriber.py --steps 600 --out models/crnn_synth.onnx

E-GMD（実データ。CC BY 4.0。GPU があれば自動で使う）::

    python train_transcriber.py --egmd D:\\data\\e-gmd-v1.0.0 --bleed-dir data\\stems --epochs 8 --out models/crnn_egmd_v1.onnx

- `--bleed-dir` にドラム以外のステムがあるフォルダを渡すと、それを混ぜて「分離の残り」を模す（ドラム系の名前のファイルは自動で除く）。
  CD 曲のステムを混ぜたモデルは評価用に限る（製品のモデルには権利の明確な音だけを使う。R3）。
- 出力: `<out>.onnx` と、特徴量の設定・種類・しきい値を書いた `<out>.json`。チェックポイント `<out>.pt` も毎エポック書く。
- 検証: E-GMD の validation 分割から `--val-tracks` 本（先頭 `--val-seconds` 秒）で種類ごとの F 値を出し、しきい値 0.3／0.4／0.5 のうち
  M0-1（kick/snare/hihat の F の平均）が最も高いものを ONNX の設定に書く。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drumeval import Hit, transcription_scores  # noqa: E402
from drumeval.features import FeatureConfig, frame_targets, log_mel, pick_hits  # noqa: E402
from drumeval.labels import M0_CLASSES  # noqa: E402
from drumeval.synth import render  # noqa: E402

THRESHOLDS = (0.3, 0.4, 0.5)


# ---------------------------------------------------------------------------
# 合成データ
# ---------------------------------------------------------------------------

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
        t = t0 + i * step + rng.normal(0, 0.004)
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


def make_example(rng: np.random.Generator, cfg: FeatureConfig, classes: tuple[str, ...], seconds: float = 6.0):
    hits = random_pattern(rng, seconds)
    x = render(hits, sr=cfg.sr, duration=seconds, bleed=float(rng.uniform(0, 0.5)),
               reverb_tail=float(rng.choice([0.0, rng.uniform(0.1, 0.6)])), seed=int(rng.integers(1 << 30)),
               variation=float(rng.uniform(0.3, 1.0)))
    x = x * 10 ** (rng.uniform(-12, 0) / 20)
    feat = log_mel(x, cfg)
    return feat, frame_targets(hits, feat.shape[0], cfg, classes), hits


def make_batch(rng: np.random.Generator, cfg: FeatureConfig, classes: tuple[str, ...], batch: int, seconds: float):
    feats, ys = [], []
    for _ in range(batch):
        f, y, _ = make_example(rng, cfg, classes, seconds)
        feats.append(f)
        ys.append(y)
    t = min(f.shape[0] for f in feats)
    return np.stack([f[:t] for f in feats]), np.stack([y[:t] for y in ys])


# ---------------------------------------------------------------------------
# 評価と書き出し（合成・E-GMD 共通）
# ---------------------------------------------------------------------------

def score_examples(predict, examples: list[tuple[np.ndarray, list[Hit]]], cfg: FeatureConfig, classes: tuple[str, ...],
                   thresholds: tuple[float, ...] = THRESHOLDS) -> dict:
    """examples: [(対数メル, 正解の打点)]。しきい値ごとに種類別の F と M0-1 を返す。"""
    probs = [predict(feat) for feat, _ in examples]
    out: dict = {}
    for thr in thresholds:
        per_class = {c: [] for c in classes}
        for prob, (_, hits) in zip(probs, examples):
            est = pick_hits(prob, cfg, classes, thr)
            s = transcription_scores(hits, est, classes)
            for c in classes:
                per_class[c].append(s[c].f_measure)
        f = {c: float(np.mean(v)) for c, v in per_class.items()}
        f["m0_1"] = float(np.mean([f[c] for c in M0_CLASSES if c in f]))
        out[thr] = f
    return out


def format_scores(scores: dict, classes: tuple[str, ...]) -> str:
    lines = []
    for thr, f in scores.items():
        lines.append(f"  thr {thr:.1f}: " + " ".join(f"{c} {f[c]:.3f}" for c in classes) + f"  M0-1 {f['m0_1']:.3f}")
    return "\n".join(lines)


def export_onnx(model, cfg: FeatureConfig, classes: tuple[str, ...], out: Path, threshold: float, meta_extra: dict) -> Path:
    import onnxruntime as ort
    import torch

    out.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    model_cpu = model.to("cpu")
    dummy = torch.zeros(1, 200, cfg.n_mels)
    torch.onnx.export(model_cpu, dummy, str(out), input_names=["log_mel"], output_names=["prob"],
                      dynamic_axes={"log_mel": {0: "batch", 1: "frames"}, "prob": {0: "batch", 1: "frames"}},
                      opset_version=17, dynamo=False)
    meta = {"features": cfg.__dict__, "classes": list(classes), "threshold": threshold} | meta_extra
    out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    # PyTorch と ONNX Runtime の出力が同じことを確かめる
    feat = np.random.default_rng(0).standard_normal((1, 300, cfg.n_mels)).astype(np.float32)
    with torch.no_grad():
        p_torch = model_cpu(torch.from_numpy(feat))[0].numpy()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    p_ort = sess.run(None, {"log_mel": feat})[0][0]
    print(f"ONNX 書き出し: {out}（{out.stat().st_size / 1024:.0f} KB）  torch と ORT の最大差 {float(np.max(np.abs(p_torch - p_ort))):.2e}")
    return out


def pick_device(name: str) -> str:
    import torch

    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# 学習（合成）
# ---------------------------------------------------------------------------

def train_synthetic(args) -> int:
    import torch
    import torch.nn.functional as F

    from drumeval.model import DrumCRNN

    classes = M0_CLASSES
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cfg = FeatureConfig()
    device = pick_device(args.device)
    model = DrumCRNN(cfg.n_mels, len(classes), hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    pos_weight = torch.tensor([8.0] * len(classes), device=device)

    def predict(feat: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            p = model(torch.from_numpy(feat[None]).to(device))[0].cpu().numpy()
        model.train()
        return p

    t_start = time.time()
    for step in range(1, args.steps + 1):
        feat, y = make_batch(rng, cfg, classes, args.batch, args.seconds)
        xb, yb = torch.from_numpy(feat).to(device), torch.from_numpy(y).to(device)
        prob = model(xb)
        loss = F.binary_cross_entropy(prob, yb, weight=1.0 + (pos_weight - 1.0) * yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            ev_rng = np.random.default_rng(1000 + step)
            examples = [(f, h) for f, _, h in (make_example(ev_rng, cfg, classes, 8.0) for _ in range(8))]
            sc = score_examples(predict, examples, cfg, classes, (0.5,))[0.5]
            print(f"step {step:5d}  loss {loss.item():.4f}  F kick {sc['kick']:.3f} snare {sc['snare']:.3f} hihat {sc['hihat']:.3f}  M0-1 {sc['m0_1']:.3f}  ({time.time() - t_start:.0f}s)", flush=True)

    export_onnx(model, cfg, classes, Path(args.out), 0.5, {"trained_on": "synthetic (drumeval.synth)", "steps": args.steps, "seed": args.seed})
    return 0


# ---------------------------------------------------------------------------
# 学習（E-GMD）
# ---------------------------------------------------------------------------

def load_eval_examples(tracks, cfg: FeatureConfig, seconds: float) -> list[tuple[np.ndarray, list[Hit]]]:
    from drumeval.egmd import load_audio
    from drumeval.labels import load_hits_midi

    examples = []
    for tr in tracks:
        x = load_audio(tr.audio, cfg.sr, 0.0, seconds)
        hits = [h for h in load_hits_midi(tr.midi) if h.time < x.size / cfg.sr]
        examples.append((log_mel(x, cfg), hits))
    return examples


def train_egmd(args) -> int:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from drumeval.egmd import CLASSES5, EGMDChunks, collect_bleed_files, read_index
    from drumeval.model import DrumCRNN

    classes = CLASSES5
    cfg = FeatureConfig()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)

    tracks = read_index(args.egmd)
    train = [t for t in tracks if t.split == "train"]
    val = [t for t in tracks if t.split == "validation"] or [t for t in tracks if t.split == "test"]
    if not train:
        print("E-GMD の train 分割が見つからない", file=sys.stderr)
        return 1
    rng.shuffle(train)  # type: ignore[arg-type]
    if args.max_tracks and len(train) > args.max_tracks:
        train = train[: args.max_tracks]
    rng.shuffle(val)  # type: ignore[arg-type]
    val = val[: args.val_tracks]
    bleed = collect_bleed_files(args.bleed_dir) if args.bleed_dir else []
    kits = sorted({t.kit for t in train})
    print(f"E-GMD: 学習 {len(train)} 本（キット {len(kits)} 種）、検証 {len(val)} 本、にじみ用の音声 {len(bleed)} 本、device={device}", flush=True)

    ds = EGMDChunks(train, cfg, chunk_sec=args.seconds, chunks_per_track=args.chunks_per_track, bleed_files=bleed,
                    do_augment=not args.no_augment, classes=classes, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                    persistent_workers=args.workers > 0)
    model = DrumCRNN(cfg.n_mels, len(classes), hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = max(len(dl) * args.epochs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)
    pos_weight = torch.tensor([8.0] * len(classes), device=device)

    def predict(feat: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            p = model(torch.from_numpy(feat[None]).to(device))[0].cpu().numpy()
        model.train()
        return p

    print("検証データを読む...", flush=True)
    val_examples = load_eval_examples(val, cfg, args.val_seconds)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best_m0, best_thr = -1.0, 0.5
    t_start = time.time()
    step = 0
    for epoch in range(1, args.epochs + 1):
        ds.set_epoch(epoch)
        model.train()
        losses = []
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            prob = model(xb)
            loss = F.binary_cross_entropy(prob, yb, weight=1.0 + (pos_weight - 1.0) * yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{total_steps}  loss {np.mean(losses[-50:]):.4f}  ({time.time() - t_start:.0f}s)", flush=True)
        scores = score_examples(predict, val_examples, cfg, classes)
        thr = max(scores, key=lambda k: scores[k]["m0_1"])
        print(f"epoch {epoch}  loss {np.mean(losses):.4f}  検証（{len(val_examples)} 本 × {args.val_seconds:.0f} 秒）:\n{format_scores(scores, classes)}", flush=True)
        torch.save({"state_dict": model.state_dict(), "hidden": args.hidden, "classes": classes, "features": cfg.__dict__, "epoch": epoch},
                   out.with_suffix(".pt"))
        if scores[thr]["m0_1"] > best_m0:
            best_m0, best_thr = scores[thr]["m0_1"], thr
            export_onnx(model, cfg, classes, out, best_thr,
                        {"trained_on": "E-GMD (CC BY 4.0)", "epochs": epoch, "train_tracks": len(train), "kits": len(kits),
                         "bleed_files": len(bleed), "val_m0_1": best_m0, "val_scores": {str(k): v for k, v in scores.items()}, "seed": args.seed})
            model.to(device)
            print(f"  → 検証の M0-1 が最良（{best_m0:.3f}、しきい値 {best_thr}）なので ONNX を更新", flush=True)
    print(f"終了。最良の検証 M0-1 {best_m0:.3f}（しきい値 {best_thr}）。モデル: {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "models" / "crnn_synth.onnx"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto / cuda / cpu")
    ap.add_argument("--hidden", type=int, default=48, help="GRU の隠れ層の大きさ（E-GMD では 96 程度を推奨）")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=6.0, help="1 区間の長さ（秒）")
    # 合成
    ap.add_argument("--steps", type=int, default=600, help="合成データの学習ステップ数")
    ap.add_argument("--eval-every", type=int, default=100)
    # E-GMD
    ap.add_argument("--egmd", default=None, help="E-GMD のフォルダ（e-gmd-v1.0.0.csv があるところ）。指定すると E-GMD で学習する")
    ap.add_argument("--bleed-dir", default=None, help="にじみ用にドラム以外のステムを混ぜるフォルダ（任意）")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-tracks", type=int, default=4000, help="学習に使うトラック数の上限（0 で全部）")
    ap.add_argument("--chunks-per-track", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2, help="DataLoader のワーカー数（Windows で問題があれば 0）")
    ap.add_argument("--val-tracks", type=int, default=40)
    ap.add_argument("--val-seconds", type=float, default=30.0)
    ap.add_argument("--no-augment", action="store_true")
    args = ap.parse_args(argv)
    if args.egmd:
        if args.hidden == 48:
            args.hidden = 96
        if args.seconds == 6.0:
            args.seconds = 8.0
        return train_egmd(args)
    return train_synthetic(args)


if __name__ == "__main__":
    raise SystemExit(main())
