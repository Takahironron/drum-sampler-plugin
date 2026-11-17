#!/usr/bin/env python3
"""プラグイン側の採譜（C++: 間引き → 対数メル → ONNX Runtime → 打点を拾う）が、Python の参照実装
（`drumeval.infer.transcribe_onnx`）と同じ結果を出すかを確かめる。

採譜モデルは Python で学習・評価する。プラグインで同じ入力（対数メル）を作れなければ、評価した精度は製品に移らない。
CI（`.github/workflows/transcribe-parity.yml`）で毎回確かめる。変えるときは Python と C++ の両方を変える。

使い方::

    python transcribe_parity.py make  作業フォルダ [--model models/crnn_synth_v0.onnx]
        # 検査用のステム（合成音）と、Python の結果（対数メル、確率、打点）を書く
    TranscribeTest models/crnn_synth_v0.onnx 作業フォルダ/mono_44100.wav 作業フォルダ/cpp_mono_44100     # 検査用のステムごとに（下の STEMS）
    ...
    python transcribe_parity.py check 作業フォルダ      # 比べる（違えば終了コード 1）
    python transcribe_parity.py run   作業フォルダ --tool <TranscribeTest の実行ファイル>      # 上の 3 つを続けて行う

- 検査用のステムは 6 本: 44.1 kHz のモノラル（2 分の 1 に間引く）、48 kHz のステレオ（147/320。チャンネルの平均）、
  22.05 kHz（間引かない）、96 kHz（147/640）、32 kHz（441/640）、16 kHz（441/320 = 補間）。長さは 1 本ずつ変える。
  32 bit float の WAV なので、Python と C++ は同じ標本を読む。
- 比べるもの: 対数メル（フレーム数と値）、モデルの出力の確率、打点（数・種類・時刻）。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drumeval import Hit  # noqa: E402
from drumeval.features import log_mel, pick_hits  # noqa: E402
from drumeval.infer import import_onnxruntime, load_meta, resample_to  # noqa: E402
from drumeval.synth import random_pattern, render  # noqa: E402
from stft_distance import load_mono  # noqa: E402

DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "crnn_synth_v0.onnx"
FEATURE_TOLERANCE = 1e-4  # 対数メル（log(1 + 1000·x)）の差の上限。倍精度で計算して float32 に丸めるので、丸めの分だけ違う
PROBABILITY_TOLERANCE = 1e-3  # 確率の差の上限。同じ版の ONNX Runtime でも、スレッドの数や CPU の命令で丸めが変わる
TIME_TOLERANCE = 1e-6  # 打点の時刻の差の上限（秒）
STEMS = ("mono_44100", "stereo_48000", "mono_22050", "mono_96000", "mono_32000", "mono_16000")


def make_stem(name: str, seconds: float, seed: int) -> tuple[np.ndarray, int]:
    """合成のドラムのステム。返り値は ([標本, チャンネル], サンプリング周波数)。"""
    kind, sr_text = name.split("_")
    sr = int(sr_text)
    rng = np.random.default_rng(seed)
    hits: list[Hit] = []
    t = 0.0
    while t < seconds - 1.0:  # テンポの違う区間をつなぐ
        seg = random_pattern(rng, min(6.0, seconds - t))
        hits += [Hit(h.time + t, h.cls, h.velocity) for h in seg]
        t += 6.5
    x = render(hits, sr=sr, duration=seconds, variation=0.6, seed=seed + 1)
    x = x + 1e-4 * rng.standard_normal(x.size)  # 録音と同じく、薄い雑音を足す
    x = 0.5 * x / (np.max(np.abs(x)) + 1e-9)
    x = x[: x.size - (seed % 97)]  # 長さを 1 本ずつ変える（間引きの端の扱いを通す）
    if kind == "stereo":  # 左右で音量と位置を変える（チャンネルの平均を確かめる）
        right = 0.7 * np.concatenate([np.zeros(7), x[:-7]])
        return np.stack([x, right], axis=1), sr
    return x[:, None], sr


def python_reference(wav: Path, model: Path) -> dict:
    ort = import_onnxruntime()

    cfg, classes, thr = load_meta(model)
    x, sr = load_mono(str(wav), None)
    if sr != cfg.sr:
        x = resample_to(x, sr, cfg.sr)
    feat = log_mel(x, cfg)
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    prob = sess.run(None, {sess.get_inputs()[0].name: feat[None]})[0][0]
    hits = pick_hits(prob, cfg, classes, thr)
    return {"features": feat, "probs": prob, "times": np.array([h.time for h in hits]), "classes": [h.cls for h in hits]}


def cmd_make(work: Path, model: Path, seconds: float, seed: int) -> int:
    import soundfile as sf

    work.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(STEMS):
        x, sr = make_stem(name, seconds if i < 3 else seconds / 3.0, seed + 10 * i + 3)
        wav = work / f"{name}.wav"
        sf.write(wav, x.astype(np.float32), sr, subtype="FLOAT")
        ref = python_reference(wav, model)
        np.savez(work / f"python_{name}.npz", features=ref["features"], probs=ref["probs"], times=ref["times"], classes=np.array(ref["classes"]))
        print(f"書き出した: {wav}（{x.shape[0] / sr:.1f} 秒、{sr} Hz、{x.shape[1]} ch）、Python: フレーム {ref['features'].shape[0]}、打 {len(ref['classes'])}")
    return 0


def compare(work: Path, name: str) -> tuple[bool, str]:
    ref = np.load(work / f"python_{name}.npz")
    cpp_dir = work / f"cpp_{name}"
    if not (cpp_dir / "info.json").exists():
        return False, f"{name}: C++ の出力が無い（{cpp_dir}）"
    info = json.loads((cpp_dir / "info.json").read_text(encoding="utf-8"))
    feat = np.fromfile(cpp_dir / "features.f32", dtype=np.float32)
    prob = np.fromfile(cpp_dir / "probs.f32", dtype=np.float32)
    problems = []
    if feat.size != info["frames"] * info["mels"] or (info["frames"], info["mels"]) != ref["features"].shape:
        return False, f"{name}: 対数メルの形が違う（C++ {info['frames']}×{info['mels']}、Python {ref['features'].shape}）"
    feat = feat.reshape(info["frames"], info["mels"])
    d_feat = float(np.max(np.abs(feat - ref["features"])))
    if d_feat > FEATURE_TOLERANCE:
        problems.append(f"対数メルの差 {d_feat:.2e} > {FEATURE_TOLERANCE:g}")
    if prob.size != ref["probs"].size:
        return False, f"{name}: 確率の形が違う（C++ {prob.size}、Python {ref['probs'].shape}）"
    prob = prob.reshape(ref["probs"].shape)
    d_prob = float(np.max(np.abs(prob - ref["probs"])))
    if d_prob > PROBABILITY_TOLERANCE:
        problems.append(f"確率の差 {d_prob:.2e} > {PROBABILITY_TOLERANCE:g}")
    with open(cpp_dir / "hits.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    times = np.array([float(r["time_sec"]) for r in rows])
    classes = [r["class"] for r in rows]
    d_time = 0.0
    if classes != list(ref["classes"]):
        problems.append(f"打点の種類の並びが違う（C++ {len(classes)} 打、Python {len(ref['classes'])} 打）")
    else:
        d_time = float(np.max(np.abs(times - ref["times"]))) if times.size else 0.0
        if d_time > TIME_TOLERANCE:
            problems.append(f"打点の時刻の差 {d_time:.2e} 秒")
    ms = info.get("milliseconds", {})
    line = (f"{name}: {info['seconds']:.1f} 秒、フレーム {info['frames']}、打 {len(classes)}、対数メルの差の最大 {d_feat:.2e}、確率の差の最大 {d_prob:.2e}、"
            f"打点の時刻の差の最大 {d_time:.1e} 秒  C++ の所要: 間引き {ms.get('resample', 0):.0f} ms、対数メル {ms.get('features', 0):.0f} ms、"
            f"推論 {ms.get('inference', 0):.0f} ms")
    return not problems, line + ("" if not problems else "  ← " + "、".join(problems))


def cmd_check(work: Path) -> int:
    ok = True
    for name in STEMS:
        good, line = compare(work, name)
        print(line)
        ok &= good
    print("一致" if ok else "不一致")
    return 0 if ok else 1


def cmd_run(work: Path, model: Path, tool: Path, seconds: float, seed: int) -> int:
    """make → C++ のプログラム → check を続けて行う。"""
    import subprocess

    if cmd_make(work, model, seconds, seed) != 0:
        return 1
    for name in STEMS:
        r = subprocess.run([str(tool), str(model), str(work / f"{name}.wav"), str(work / f"cpp_{name}")])
        if r.returncode != 0:
            print(f"{tool.name} が失敗した（{name}、終了コード {r.returncode}）")
            return 1
    return cmd_check(work)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("make", "check", "run"))
    ap.add_argument("work_dir", help="検査用のステムと、Python・C++ の結果を置くフォルダ")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="ONNX の採譜モデル（同じ名前の .json が要る）")
    ap.add_argument("--seconds", type=float, default=30.0, help="検査用のステムの長さ（make）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tool", default=None, help="TranscribeTest の実行ファイル（run）")
    args = ap.parse_args(argv)
    work = Path(args.work_dir)
    if args.command == "make":
        return cmd_make(work, Path(args.model), args.seconds, args.seed)
    if args.command == "run":
        if not args.tool:
            ap.error("run には --tool が要る")
        return cmd_run(work, Path(args.model), Path(args.tool), args.seconds, args.seed)
    return cmd_check(work)


if __name__ == "__main__":
    raise SystemExit(main())
