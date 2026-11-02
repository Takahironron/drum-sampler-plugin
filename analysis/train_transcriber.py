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
  エポックごとの検証の結果は `<out>.log.txt` に残る。
- 検証: E-GMD の validation 分割から `--val-tracks` 本（先頭 `--val-seconds` 秒）で種類ごとの F 値を出し、しきい値 0.3〜0.8 のうち
  種類ごとに F が最も高いものを選んで ONNX の設定（`thresholds`）に書く。M0-1（kick/snare/hihat の F の平均）が最良のエポックを残す。
- 特徴量と形: 64 メル帯域、周波数方向の縮小 2 回（v3 で 96 メル・縮小 1 回を試したが効かなかった）。v4 から、対数メルの時間差分を
  2 つ目のチャンネルとしてモデルの中で作る（小さい打の立ち上がりを拾う）。
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

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


# ---------------------------------------------------------------------------
# 合成データ
# ---------------------------------------------------------------------------

from drumeval.synth import random_pattern  # noqa: E402


def make_example(rng: np.random.Generator, cfg: FeatureConfig, classes: tuple[str, ...], seconds: float = 6.0):
    hits = random_pattern(rng, seconds, five_classes=len(classes) > 3)
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

def best_thresholds(scores: dict, classes: tuple[str, ...]) -> dict[str, float]:
    """しきい値ごとの結果から、種類ごとに F が最大のしきい値を選ぶ。"""
    return {c: float(max(scores, key=lambda k: scores[k][c])) for c in classes}


def score_with_thresholds(predict, examples, cfg: FeatureConfig, classes: tuple[str, ...], thr: dict[str, float]) -> dict:
    """種類別のしきい値で採点する。"""
    per_class = {c: [] for c in classes}
    for feat, hits in examples:
        est = pick_hits(predict(feat), cfg, classes, thr)
        s = transcription_scores(hits, est, classes)
        for c in classes:
            per_class[c].append(s[c].f_measure)
    f = {c: float(np.mean(v)) for c, v in per_class.items()}
    f["m0_1"] = float(np.mean([f[c] for c in M0_CLASSES if c in f]))
    return f


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


def export_onnx(model, cfg: FeatureConfig, classes: tuple[str, ...], out: Path, threshold: float | dict[str, float], meta_extra: dict) -> Path:
    import onnxruntime as ort
    import torch

    out.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    model_cpu = model.to("cpu")
    dummy = torch.zeros(1, 200, cfg.n_mels)
    torch.onnx.export(model_cpu, dummy, str(out), input_names=["log_mel"], output_names=["prob"],
                      dynamic_axes={"log_mel": {0: "batch", 1: "frames"}, "prob": {0: "batch", 1: "frames"}},
                      opset_version=17, dynamo=False)
    meta = {"features": cfg.__dict__, "classes": list(classes)} | meta_extra
    if isinstance(threshold, dict):
        meta["thresholds"] = threshold
        meta["threshold"] = float(np.mean(list(threshold.values())))
    else:
        meta["threshold"] = threshold
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
    cfg = FeatureConfig(n_mels=64)
    device = pick_device(args.device)
    model = DrumCRNN(cfg.n_mels, len(classes), hidden=args.hidden, pools=2).to(device)
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
    cfg = FeatureConfig(n_mels=args.n_mels)
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
                    do_augment=not args.no_augment, classes=classes, seed=args.seed, synthetic_ratio=args.synthetic_ratio)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                    persistent_workers=args.workers > 0)
    model = DrumCRNN(cfg.n_mels, len(classes), hidden=args.hidden, pools=args.pools, delta=not args.no_delta, layers=args.layers).to(device)
    # 重みの移動平均（EMA）。エポック間の揺れを抑え、検証と test の差を小さくする
    import copy

    ema = copy.deepcopy(model).eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    ema_decay = args.ema

    def update_ema() -> None:
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(ema_decay).add_(pm.detach(), alpha=1.0 - ema_decay)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = max(len(dl) * args.epochs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)
    # 打点のフレームは少ないので重みを付ける。ハイハットとスネアは小さい打の取りこぼしが多いので重く、シンバルは誤検出が多いので軽く
    pos_weight = torch.tensor([6.0, 8.0, 8.0, 8.0, 4.0][: len(classes)], device=device)

    def make_predict(net):
        def predict(feat: np.ndarray) -> np.ndarray:
            net.eval()
            with torch.no_grad():
                p = net(torch.from_numpy(feat[None]).to(device))[0].cpu().numpy()
            return p
        return predict

    predict_raw = make_predict(model)
    predict_ema = make_predict(ema)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(".log.txt")
    log_path.write_text(f"E-GMD: 学習 {len(train)} 本（キット {len(kits)} 種）、検証 {len(val)} 本、にじみ用の音声 {len(bleed)} 本、"
                        f"合成の混ぜ込み {args.synthetic_ratio:g}、hidden {args.hidden}、layers {args.layers}、n_mels {args.n_mels}、pools {args.pools}、"
                        f"delta {not args.no_delta}、EMA {args.ema}、batch {args.batch}、epochs {args.epochs}、lr {args.lr}\n", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    print("検証データを読む...", flush=True)
    val_examples = load_eval_examples(val, cfg, args.val_seconds)
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
            update_ema()
            losses.append(loss.item())
            step += 1
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{total_steps}  loss {np.mean(losses[-50:]):.4f}  ({time.time() - t_start:.0f}s)", flush=True)
        # 生の重みと EMA の重みの両方を検証し、良い方を残す
        cands = {}
        for name, pred in (("raw", predict_raw), ("ema", predict_ema)):
            scores = score_examples(pred, val_examples, cfg, classes)
            thr_by_class = best_thresholds(scores, classes)
            f_best = score_with_thresholds(pred, val_examples, cfg, classes, thr_by_class)
            cands[name] = (scores, thr_by_class, f_best)
        model.train()
        best_name = max(cands, key=lambda k: cands[k][2]["m0_1"])
        scores, thr_by_class, f_best = cands[best_name]
        log(f"epoch {epoch}  loss {np.mean(losses):.4f}  検証（{len(val_examples)} 本 × {args.val_seconds:.0f} 秒）  ({time.time() - t_start:.0f}s):\n{format_scores(scores, classes)}\n"
            f"  種類別のしきい値 {thr_by_class}: " + " ".join(f"{c} {f_best[c]:.3f}" for c in classes) + f"  M0-1 {f_best['m0_1']:.3f}"
            f"（raw {cands['raw'][2]['m0_1']:.3f} / ema {cands['ema'][2]['m0_1']:.3f} → {best_name}）")
        torch.save({"state_dict": model.state_dict(), "ema_state_dict": ema.state_dict(), "hidden": args.hidden, "layers": args.layers, "pools": args.pools,
                    "delta": not args.no_delta, "classes": classes, "features": cfg.__dict__, "epoch": epoch}, out.with_suffix(".pt"))
        if f_best["m0_1"] > best_m0:
            best_m0, best_thr = f_best["m0_1"], thr_by_class
            chosen = ema if best_name == "ema" else model
            export_onnx(chosen, cfg, classes, out, best_thr,
                        {"trained_on": "E-GMD (CC BY 4.0)" + (f" + synthetic {args.synthetic_ratio:g}" if args.synthetic_ratio else ""),
                         "epochs": epoch, "train_tracks": len(train), "kits": len(kits), "hidden": args.hidden, "layers": args.layers, "pools": args.pools,
                         "delta": not args.no_delta, "weights": best_name, "ema": args.ema,
                         "bleed_files": len(bleed), "val_m0_1": best_m0, "val_scores": {str(k): v for k, v in scores.items()}, "seed": args.seed})
            model.to(device)
    log(f"終了。最良の検証 M0-1 {best_m0:.3f}（しきい値 {best_thr}）。モデル: {out}。ログ: {log_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "models" / "crnn_synth.onnx"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto / cuda / cpu")
    ap.add_argument("--hidden", type=int, default=48, help="GRU の隠れ層の大きさ（E-GMD では既定 160）")
    ap.add_argument("--layers", type=int, default=1, help="GRU の層数（E-GMD では既定 2）")
    ap.add_argument("--ema", type=float, default=0.999, help="重みの移動平均の減衰（E-GMD）。0 で使わない")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=6.0, help="1 区間の長さ（秒）")
    # 合成
    ap.add_argument("--steps", type=int, default=600, help="合成データの学習ステップ数")
    ap.add_argument("--eval-every", type=int, default=100)
    # E-GMD
    ap.add_argument("--egmd", default=None, help="E-GMD のフォルダ（e-gmd-v1.0.0.csv があるところ）。指定すると E-GMD で学習する")
    ap.add_argument("--bleed-dir", default=None, help="にじみ用にドラム以外のステムを混ぜるフォルダ（任意）")
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-tracks", type=int, default=8000, help="学習に使うトラック数の上限（0 で全部。v4 から既定 8000）")
    ap.add_argument("--chunks-per-track", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4, help="DataLoader のワーカー数（Windows で問題があれば 0）")
    ap.add_argument("--val-tracks", type=int, default=160)
    ap.add_argument("--n-mels", type=int, default=64, help="メル帯域の数（既定 64。v3 で 96 を試したが効かなかった）")
    ap.add_argument("--pools", type=int, default=2, choices=(1, 2), help="周波数方向に半分にする回数（既定 2）")
    ap.add_argument("--no-delta", action="store_true", help="対数メルの時間差分のチャンネルを使わない（v4 から既定で使う）")
    ap.add_argument("--val-seconds", type=float, default=30.0)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--synthetic-ratio", type=float, default=0.15, help="E-GMD の区間の代わりに合成音（電子系のキックを含む）を返す割合（E-GMD のとき）")
    args = ap.parse_args(argv)
    if args.egmd:
        if args.hidden == 48:
            args.hidden = 160
        if args.layers == 1:
            args.layers = 2
        if args.seconds == 6.0:
            args.seconds = 8.0
        return train_egmd(args)
    return train_synthetic(args)


if __name__ == "__main__":
    raise SystemExit(main())
