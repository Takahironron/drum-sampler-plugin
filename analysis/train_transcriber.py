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
微調整（実ステムの疑似ラベル。評価用ステムに合わせる。leave-one-out で曲ごとに前後を測る）::

    python train_transcriber.py --finetune-stems data\\stems --init-from models/crnn_egmd_v5.pt --holdout all --egmd D:\\data\\e-gmd-v1.0.0 --out models/crnn_ft_v5.onnx

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
from datetime import date
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
        ckpt = {"state_dict": model.state_dict(), "ema_state_dict": ema.state_dict(), "hidden": args.hidden, "layers": args.layers, "pools": args.pools,
                "delta": not args.no_delta, "classes": classes, "features": cfg.__dict__, "epoch": epoch, "weights": best_name, "thresholds": thr_by_class}
        torch.save(ckpt, out.with_suffix(".pt"))  # 最終エポック
        if f_best["m0_1"] > best_m0:
            best_m0, best_thr = f_best["m0_1"], thr_by_class
            chosen = ema if best_name == "ema" else model
            torch.save(ckpt, out.with_suffix(".best.pt"))  # 最良エポック（ONNX と同じ重み。微調整の初期値に使う）
            export_onnx(chosen, cfg, classes, out, best_thr,
                        {"trained_on": "E-GMD (CC BY 4.0)" + (f" + synthetic {args.synthetic_ratio:g}" if args.synthetic_ratio else ""),
                         "epochs": epoch, "train_tracks": len(train), "kits": len(kits), "hidden": args.hidden, "layers": args.layers, "pools": args.pools,
                         "delta": not args.no_delta, "weights": best_name, "ema": args.ema,
                         "bleed_files": len(bleed), "val_m0_1": best_m0, "val_scores": {str(k): v for k, v in scores.items()}, "seed": args.seed})
            model.to(device)
    log(f"終了。最良の検証 M0-1 {best_m0:.3f}（しきい値 {best_thr}）。モデル: {out}。ログ: {log_path}")
    return 0


# ---------------------------------------------------------------------------
# 微調整（実ステムの疑似ラベル、leave-one-out）
# ---------------------------------------------------------------------------

def load_checkpoint(path: str | Path, device: str, weights: str = "auto"):
    """`train_egmd` のチェックポイント（.pt / .best.pt）からモデルを作る。`weights`: raw / ema / auto（学習時に選ばれた方。無ければ raw）。"""
    import torch

    from drumeval.model import DrumCRNN

    ck = torch.load(str(path), map_location=device)
    cfg = FeatureConfig(**ck["features"])
    classes = tuple(ck["classes"])
    model = DrumCRNN(cfg.n_mels, len(classes), hidden=ck["hidden"], pools=ck.get("pools", 2), delta=ck.get("delta", True), layers=ck.get("layers", 1))
    if weights == "auto":
        weights = ck.get("weights", "raw")
    model.load_state_dict(ck["ema_state_dict"] if weights == "ema" and "ema_state_dict" in ck else ck["state_dict"])
    return model.to(device), cfg, classes, ck, weights


def score_songs(predict, songs, feats, cfg: FeatureConfig, classes: tuple[str, ...], thresholds=THRESHOLDS) -> dict:
    """曲ごとの採点。種類 c の F は、c のラベルがある曲だけで平均する（ラベルの無い種類は None）。
    返り値: {しきい値: {種類: F か None, "m0_1": …}}。しきい値は種類ごとに別でもよい（辞書）。"""
    probs = [predict(f) for f in feats]
    out: dict = {}
    for thr in thresholds:
        per_class: dict[str, list[float]] = {c: [] for c in classes}
        for prob, song in zip(probs, songs):
            est = pick_hits(prob, cfg, classes, thr)
            sc = transcription_scores(song.hits, est, classes)
            for c in classes:
                if c in song.labeled:
                    per_class[c].append(sc[c].f_measure)
        f = {c: (float(np.mean(v)) if v else None) for c, v in per_class.items()}
        m0 = [f[c] for c in M0_CLASSES if f.get(c) is not None]
        f["m0_1"] = float(np.mean(m0)) if m0 else None
        out[thr if not isinstance(thr, dict) else "by_class"] = f
    return out


def score_at(predict, songs, feats, cfg: FeatureConfig, classes: tuple[str, ...], thr: float | dict[str, float]) -> dict:
    """1 つのしきい値（1 つの値か種類別の辞書）での曲ごとの採点。"""
    return next(iter(score_songs(predict, songs, feats, cfg, classes, (thr,)).values()))


def pick_thresholds(scores: dict, classes: tuple[str, ...], fallback: dict[str, float] | float) -> dict[str, float]:
    """種類ごとに F が最大のしきい値。ラベルの無い種類は `fallback`（チェックポイントのしきい値）。"""
    out = {}
    for c in classes:
        cands = [(scores[t][c], t) for t in scores if scores[t][c] is not None]
        if cands:
            out[c] = float(max(cands)[1])
        else:
            out[c] = float(fallback.get(c, 0.5)) if isinstance(fallback, dict) else float(fallback)
    return out


def _fmt_f(f: dict | None, classes=M0_CLASSES) -> str:
    if not f:
        return "-"
    return " ".join(f"{c} {f[c]:.3f}" if f.get(c) is not None else f"{c} -" for c in classes) + (f"  M0-1 {f['m0_1']:.3f}" if f.get("m0_1") is not None else "")


class _WithMask:
    """E-GMD の区間に「全種類ラベルあり」のマスクを付ける（`StemChunks` と同じ形にする）。"""

    def __init__(self, ds, n_classes: int):
        import torch

        self.ds = ds
        self.mask = torch.ones(n_classes)

    def set_epoch(self, epoch: int) -> None:
        self.ds.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        feat, y = self.ds[i]
        return feat, y, self.mask


def finetune_one(args, songs, holdout_id: str | None, egmd_train, egmd_val, bleed, device: str, log) -> dict:
    """`holdout_id` 以外の曲で微調整し、`holdout_id` で前後を測る。返り値: 前後の採点と出力先。"""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import ConcatDataset, DataLoader

    from drumeval.egmd import EGMDChunks, estimate_offset, load_audio
    from drumeval.stems import StemChunks, song_examples

    train_songs = [s for s in songs if s.song_id != holdout_id]
    hold_songs = [s for s in songs if s.song_id == holdout_id]
    model, cfg, classes, ck, weights = load_checkpoint(args.init_from, device, args.init_weights)
    ck_thr = ck.get("thresholds")
    if ck_thr is None:  # 古いチェックポイント（v5 まで）はしきい値を持たないので、横の json（ONNX の設定）から取る
        meta_path = Path(args.init_from).with_name(Path(args.init_from).name.split(".")[0] + ".json")  # crnn_egmd_v5.best.pt → crnn_egmd_v5.json
        if meta_path.exists():
            ck_thr = json.loads(meta_path.read_text(encoding="utf-8")).get("thresholds")
    ck_thr = ck_thr or 0.5
    torch.manual_seed(args.seed)

    train_feats = [f for f, _ in song_examples(train_songs, cfg)]
    hold_feats = [f for f, _ in song_examples(hold_songs, cfg)]

    def make_predict(net):
        def predict(feat: np.ndarray) -> np.ndarray:
            net.eval()
            with torch.no_grad():
                p = net(torch.from_numpy(feat[None]).to(device))[0].cpu().numpy()
            net.train()
            return p
        return predict

    predict = make_predict(model)

    def evaluate() -> dict:
        sc_train = score_songs(predict, train_songs, train_feats, cfg, classes)
        thr = pick_thresholds(sc_train, classes, ck_thr)
        r = {"thresholds": thr, "train": score_at(predict, train_songs, train_feats, cfg, classes, thr)}
        if hold_songs:
            r["holdout"] = score_at(predict, hold_songs, hold_feats, cfg, classes, thr)
            r["holdout_ckpt_thr"] = score_at(predict, hold_songs, hold_feats, cfg, classes, ck_thr)
        if egmd_val:
            r["egmd"] = score_with_thresholds(predict, egmd_val, cfg, classes, thr)
            r["egmd_ckpt_thr"] = score_with_thresholds(predict, egmd_val, cfg, classes, ck_thr)  # 初期値のしきい値（しきい値の違いと忘れ具合を分ける）
        return r

    tag = holdout_id or "all"
    log(f"[{tag}] 学習 {len(train_songs)} 曲（{', '.join(s.song_id for s in train_songs)}）"
        + (f"、評価 {holdout_id}" if holdout_id else "、評価は学習した曲（当てはまりだけ）")
        + f"、初期値 {Path(args.init_from).name}（{weights}）、E-GMD の混ぜ込み {len(egmd_train)} 本、device={device}")
    # ラベルと音声のずれの見積もり（分離ツールのパートの打点検出が遅れていないか）
    offs = []
    for s_ in train_songs[:10]:
        x = load_audio(s_.audio, cfg.sr, s_.start, min(s_.seconds, 30.0))
        offs.append(f"{s_.song_id} {estimate_offset(x, cfg.sr, [h for h in s_.hits if h.time < x.size / cfg.sr], cfg) * 1000:+.0f} ms")
    log(f"[{tag}] ラベルのずれの見積もり（正なら音声が遅い）: " + ", ".join(offs))
    before = evaluate()
    log(f"[{tag}] 前: 学習曲 {_fmt_f(before['train'])}" + (f" / 評価曲 {_fmt_f(before.get('holdout'))}（初期値のしきい値では {_fmt_f(before.get('holdout_ckpt_thr'))}）" if hold_songs else "")
        + (f" / E-GMD 検証 M0-1 {before['egmd']['m0_1']:.3f}（初期値のしきい値では {before['egmd_ckpt_thr']['m0_1']:.3f}）" if egmd_val else "") + f"  しきい値 {before['thresholds']}")
    import copy

    init_state = copy.deepcopy(model.state_dict()) if args.mix_init > 0 else None

    ds_stems = StemChunks(train_songs, cfg, chunk_sec=args.seconds, chunks_per_song=args.chunks_per_song, bleed_files=bleed,
                          do_augment=not args.no_augment, classes=classes, seed=args.seed)
    parts = [ds_stems]
    if egmd_train:
        parts.append(_WithMask(EGMDChunks(egmd_train, cfg, chunk_sec=args.seconds, chunks_per_track=args.chunks_per_track, bleed_files=bleed,
                                          do_augment=not args.no_augment, classes=classes, seed=args.seed, synthetic_ratio=args.synthetic_ratio), len(classes)))
    dl = DataLoader(ConcatDataset(parts), batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                    persistent_workers=args.workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = max(len(dl) * args.epochs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.3)
    pos_weight = torch.tensor([6.0, 8.0, 8.0, 8.0, 4.0][: len(classes)], device=device)
    t_start = time.time()
    step = 0
    for epoch in range(1, args.epochs + 1):
        for d in parts:
            d.set_epoch(epoch)
        model.train()
        losses = []
        for xb, yb, mb in dl:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            prob = model(xb)
            w = (1.0 + (pos_weight - 1.0) * yb) * mb[:, None, :]  # ラベルの無い種類は損失に入れない
            loss = (F.binary_cross_entropy(prob, yb, reduction="none") * w).sum() / w.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
        r = evaluate()
        log(f"[{tag}] epoch {epoch}/{args.epochs}  loss {np.mean(losses):.4f}  学習曲 {_fmt_f(r['train'])}"
            + (f" / 評価曲 {_fmt_f(r.get('holdout'))}" if hold_songs else "") + (f" / E-GMD 検証 M0-1 {r['egmd']['m0_1']:.3f}" if egmd_val else "")
            + f"  ({time.time() - t_start:.0f}s)")
    if init_state is not None:
        # 初期値との重みの混合（robust fine-tuning。学習した曲に寄り過ぎるのを抑え、初期値が良かった曲の性能を残す）
        with torch.no_grad():
            sd = model.state_dict()
            for k, v in sd.items():
                if v.is_floating_point():
                    v.copy_(args.mix_init * init_state[k].to(v.device) + (1.0 - args.mix_init) * v)
        log(f"[{tag}] 初期値の重みを {args.mix_init:g} の割合で混ぜた")
    after = evaluate()
    out = Path(args.out)
    if holdout_id:
        out = out.with_name(f"{out.stem}_{holdout_id}{out.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(model, cfg, classes, out, after["thresholds"],
                {"trained_on": "E-GMD (CC BY 4.0) + fine-tuned on evaluation stems with pseudo labels (evaluation only; R3)",
                 "init_from": str(Path(args.init_from).name), "init_weights": weights, "holdout": holdout_id,
                 "train_songs": [s.song_id for s in train_songs], "epochs": args.epochs, "lr": args.lr, "chunks_per_song": args.chunks_per_song, "mix_init": args.mix_init,
                 "egmd_tracks": len(egmd_train), "before": {k: v for k, v in before.items() if k != "thresholds"},
                 "after": {k: v for k, v in after.items() if k != "thresholds"}, "seed": args.seed})
    torch.save({"state_dict": model.state_dict(), "hidden": ck["hidden"], "layers": ck.get("layers", 1), "pools": ck.get("pools", 2), "delta": ck.get("delta", True),
                "classes": classes, "features": cfg.__dict__, "epoch": args.epochs, "weights": "raw", "thresholds": after["thresholds"],
                "init_from": str(args.init_from), "holdout": holdout_id}, out.with_suffix(".pt"))
    return {"holdout": holdout_id, "before": before, "after": after, "out": out}


def finetune_stems(args) -> int:
    """`--finetune-stems`: 実ステムの疑似ラベル（`run_eval.py` が書く hits_pseudo.csv）で、E-GMD のモデルを短く学習し直す。

    `--holdout all` なら曲ごとに「その曲以外で学習 → その曲で測る」を繰り返し（leave-one-out）、曲ごとのモデル
    `<out>_<曲ID>.onnx` と前後の表 `<out>_loo.md` を書く。`run_eval.py --model "<out>_{id}.onnx"` で曲ごとに使える。
    """
    from drumeval.egmd import collect_bleed_files, read_index
    from drumeval.stems import read_stem_songs

    device = pick_device(args.device)
    ids = [t.strip() for t in args.song_ids.split(",") if t.strip()] if args.song_ids else None
    songs = read_stem_songs(args.finetune_stems, ids, max_seconds=args.max_seconds)
    if len(songs) < 2:
        print(f"疑似正解（hits_pseudo.csv）か正解のある曲が 2 曲以上要る（{args.finetune_stems}: {len(songs)} 曲）。先に run_eval.py を走らせる", file=sys.stderr)
        return 1
    rng = np.random.default_rng(args.seed)
    egmd_train, egmd_val = [], []
    if args.egmd:
        import torch

        tracks = read_index(args.egmd)
        tr = [t for t in tracks if t.split == "train"]
        va = [t for t in tracks if t.split == "validation"] or [t for t in tracks if t.split == "test"]
        rng.shuffle(tr)  # type: ignore[arg-type]
        rng.shuffle(va)  # type: ignore[arg-type]
        egmd_train = tr[: args.max_tracks] if args.max_tracks else tr
        print("E-GMD の検証データを読む...", flush=True)
        cfg0 = FeatureConfig(**torch.load(str(args.init_from), map_location="cpu")["features"])
        egmd_val = load_eval_examples(va[: args.val_tracks], cfg0, args.val_seconds)
    bleed = collect_bleed_files(args.bleed_dir) if args.bleed_dir else []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(".log.txt")
    log_path.write_text(f"微調整: {args.finetune_stems} の {len(songs)} 曲（{', '.join(s.song_id for s in songs)}）、初期値 {args.init_from}、"
                        f"holdout {args.holdout or 'なし'}、epochs {args.epochs}、lr {args.lr}、chunks_per_song {args.chunks_per_song}、"
                        f"E-GMD {len(egmd_train)} 本、にじみ {len(bleed)} 本\n", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    for s_ in songs:
        log(f"  {s_.song_id}: {s_.audio.name}、開始 {s_.start:.1f} 秒、{s_.seconds:.0f} 秒、打 {len(s_.hits)}、ラベルのある種類 {','.join(s_.labeled)}（{s_.truth}）")
    if args.holdout == "all":
        holdouts: list[str | None] = [s_.song_id for s_ in songs]
    elif args.holdout:
        holdouts = [t.strip() for t in args.holdout.split(",") if t.strip()]
        unknown = [h for h in holdouts if h not in {s_.song_id for s_ in songs}]
        if unknown:
            print(f"--holdout に無い曲 ID: {unknown}", file=sys.stderr)
            return 1
    else:
        holdouts = [None]
    results = [finetune_one(args, songs, h, egmd_train, egmd_val, bleed, device, log) for h in holdouts]

    # まとめの表（leave-one-out なら評価曲の前後。holdout 無しなら学習曲の当てはまり）
    lines = [f"# 微調整の結果（{date.today().isoformat()}。初期値 {Path(args.init_from).name}、{len(songs)} 曲、epochs {args.epochs}、lr {args.lr}）", ""]
    key = "holdout" if holdouts != [None] else "train"
    n_extra = 2 if egmd_val else 0
    head = ("| 曲ID | 前（初期値のしきい値）M0-1 | 前 kick | 前 snare | 前 hihat | 前 M0-1 | 後 kick | 後 snare | 後 hihat | 後 M0-1 | 差 |"
            + (" E-GMD 検証 前→後 | E-GMD 検証（初期値のしきい値）前→後 |" if egmd_val else ""))
    lines += [head, "|---" * (11 + n_extra) + "|"]
    sums = {"before0": [], "before": [], "after": []}
    for r in results:
        b, a = r["before"][key], r["after"][key]
        b0 = r["before"].get("holdout_ckpt_thr") if key == "holdout" else None
        if b.get("m0_1") is not None and a.get("m0_1") is not None:
            sums["before"].append(b["m0_1"])
            sums["after"].append(a["m0_1"])
            if b0 and b0.get("m0_1") is not None:
                sums["before0"].append(b0["m0_1"])
        cells = [r["holdout"] or "（学習曲の当てはまり）", f"{b0['m0_1']:.3f}" if b0 and b0.get("m0_1") is not None else "-"]
        cells += [f"{b[c]:.3f}" if b.get(c) is not None else "-" for c in M0_CLASSES] + [f"{b['m0_1']:.3f}" if b.get("m0_1") is not None else "-"]
        cells += [f"{a[c]:.3f}" if a.get(c) is not None else "-" for c in M0_CLASSES] + [f"{a['m0_1']:.3f}" if a.get("m0_1") is not None else "-"]
        cells.append(f"{a['m0_1'] - b['m0_1']:+.3f}" if b.get("m0_1") is not None and a.get("m0_1") is not None else "-")
        if egmd_val:
            cells.append(f"{r['before']['egmd']['m0_1']:.3f} → {r['after']['egmd']['m0_1']:.3f}")
            cells.append(f"{r['before']['egmd_ckpt_thr']['m0_1']:.3f} → {r['after']['egmd_ckpt_thr']['m0_1']:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    if sums["before"]:
        b0_avg = f"{np.mean(sums['before0']):.3f}" if sums["before0"] else ""
        lines.append(f"| 平均 | {b0_avg} | | | | {np.mean(sums['before']):.3f} | | | | {np.mean(sums['after']):.3f} | {np.mean(sums['after']) - np.mean(sums['before']):+.3f} |"
                     + (" | |" if egmd_val else ""))
    lines += ["",
              "- 「前」は初期値のモデル、「後」は微調整したモデル。しきい値はどちらも学習した曲で種類ごとに選ぶ（評価曲は見ない）。「前（初期値のしきい値）」は初期値のモデルを初期値の json のしきい値で測ったもの（run_eval.py の表と同じ条件）。"
              + (f" 微調整の後に初期値の重みを {args.mix_init:g} の割合で混ぜた（--mix-init）。" if args.mix_init > 0 else ""),
              "- 疑似正解（分離ツールのパート別ステムの打点検出）に対する F 値なので目安。" + ("評価曲はその曲を学習から外したモデル（leave-one-out）で測った。" if key == "holdout" else "holdout 無しなので当てはまりの数字で、汎化の目安にはならない。"),
              f"- 曲ごとのモデル: `{Path(args.out).stem}_<曲ID>.onnx`。`run_eval.py --model \"{Path(args.out).with_suffix('')}_{{id}}.onnx\"` で曲ごとに使う。" if key == "holdout" else f"- モデル: `{args.out}`。",
              "- CD 曲のステムで学習したモデルは評価用に限る（製品のモデルには権利の明確な音だけを使う。R3）。"]
    summary = "\n".join(lines) + "\n"
    summary_path = out.with_name(f"{out.stem}_loo.md" if key == "holdout" else f"{out.stem}_fit.md")
    summary_path.write_text(summary, encoding="utf-8")
    print(summary)
    log(f"終了。まとめ: {summary_path}。ログ: {log_path}")
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
    # 微調整（実ステムの疑似ラベル）
    ap.add_argument("--finetune-stems", default=None, metavar="DIR",
                    help="評価用ステムのフォルダ（data/stems）。run_eval.py が書いた hits_pseudo.csv（か正解の hits.csv／.mid）のある曲で、"
                         "--init-from のモデルを短く学習し直す。--egmd を併せて渡すと E-GMD の区間も混ぜる（忘れ防止。--max-tracks 本、既定 200）")
    ap.add_argument("--init-from", default=None, help="初期値のチェックポイント（train_egmd の .best.pt か .pt）。微調整では必須")
    ap.add_argument("--init-weights", default="auto", choices=("auto", "raw", "ema"), help="チェックポイントの生の重みか EMA か（auto は学習時に選ばれた方）")
    ap.add_argument("--holdout", default=None, help="評価に回す曲 ID（コンマ区切り）。all なら曲ごとに leave-one-out。省略すると全曲で学習（当てはまりだけ出る）")
    ap.add_argument("--song-ids", default=None, help="微調整に使う曲 ID をコンマ区切りで絞る（省略で全曲）")
    ap.add_argument("--chunks-per-song", type=int, default=40, help="1 曲から 1 エポックに切り出す区間の数（微調整）")
    ap.add_argument("--max-seconds", type=float, default=60.0, help="古い hits_pseudo.csv（コメント行なし）の区間の長さ（run_eval.py の既定と同じ）")
    ap.add_argument("--mix-init", type=float, default=0.0,
                    help="微調整の後に初期値の重みを混ぜる割合（0〜1。0.5 で半々）。学習した曲に寄り過ぎるのを抑える（robust fine-tuning）。既定 0")
    args = ap.parse_args(argv)
    if args.finetune_stems:
        if not args.init_from:
            print("--finetune-stems には --init-from（初期値の .pt）が要る", file=sys.stderr)
            return 2
        # 微調整の既定: 短く・小さい学習率・EMA なし・E-GMD は少なめ
        if args.epochs == 16:
            args.epochs = 6
        if args.lr == 1e-3:
            args.lr = 1e-4
        if args.ema == 0.999:
            args.ema = 0.0
        if args.max_tracks == 8000:
            args.max_tracks = 200  # × chunks_per_track 2 = 400 区間。実ステム 9 曲 × 40 区間と同じくらい
        if args.val_tracks == 160:
            args.val_tracks = 40
        if args.seconds == 6.0:
            args.seconds = 8.0
        return finetune_stems(args)
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
