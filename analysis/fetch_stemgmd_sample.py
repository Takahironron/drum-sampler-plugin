#!/usr/bin/env python3
"""StemGMD（CC BY 4.0）の評価用セッションから、パート別の単独録音が付いた少数の曲を取ってくる。

StemGMD は Groove MIDI Dataset の演奏を 10 種類のアコースティックキットで鳴らし、9 つのパート（キック、スネア、タム 3 つ、
ハイハットの開閉、クラッシュ、ライド）を別々に書き出したもの。パート分け（NMFD）の**本当の正解**として使える::

    python analysis/fetch_stemgmd_sample.py --n 12 --out data/stemgmd_stems
    python analysis/run_eval.py data/stemgmd_stems --parts nmfd-oracle --out data/results/stemgmd_sample.md

- 取るのは評価用の zip（5 GB）の中の必要なファイルだけ（HTTP の範囲指定。1 曲 13 MB ほど）。打点の正解は Groove MIDI Dataset の MIDI。
- 置き方（`run_eval.py` の形）: `<曲ID>/drums.wav`（9 パートの和）、`hits.mid`、`kick.wav`／`snare.wav`／`hihat.wav`／`tom.wav`／`cymbal.wav`
  （粗い 5 種にまとめた本当のパート）、`start.txt`（0）。モノラル 44.1 kHz 16 bit、先頭 `--seconds` 秒。
- 選び方: キット 10 種を 1 つずつ、曲（10 種）とドラマー（4 人）を順に回す。`--n` が 10 を超えたら 2 周目。
- 出典の表示: Mezza, Giampiccolo, Zanoni, Bernardini, Sarti (2024) "Toward Deep Drum Source Separation", Pattern Recognition Letters。
  https://zenodo.org/records/7882857 （CC BY 4.0）。MIDI は Groove MIDI Dataset（Gillick et al. 2019、CC BY 4.0）。
- `data/` は `.gitignore` 済み。音声はコミットしない。
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_egmd_sample import HttpRangeFile  # noqa: E402

STEMS_URL = "https://zenodo.org/api/records/7882857/files/StemGMD_eval_session.zip/content"
MIDI_URL = "https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip"

# StemGMD の 9 パート → 粗い 5 種
STEM_TO_CLASS = {
    "kick": "kick", "snare": "snare",
    "hihat_closed": "hihat", "hihat_open": "hihat",
    "hi_tom": "tom", "mid_tom": "tom", "low_tom": "tom",
    "crash_left": "cymbal", "ride": "cymbal",
}
CLASSES = ("kick", "snare", "hihat", "tom", "cymbal")


def read_wav(z: zipfile.ZipFile, name: str) -> tuple[np.ndarray, int]:
    import soundfile as sf

    with z.open(name) as f:
        x, sr = sf.read(io.BytesIO(f.read()), dtype="float64", always_2d=True)
    return x.mean(axis=1), sr


def main(argv: list[str] | None = None) -> int:
    import soundfile as sf

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--out", default=str(root / "data" / "stemgmd_stems"), help="run_eval.py の形で置くフォルダ")
    ap.add_argument("--n", type=int, default=12, help="取ってくる曲の数")
    ap.add_argument("--seconds", type=float, default=40.0, help="先頭から何秒を使うか（0 で全部）")
    ap.add_argument("--offset", type=int, default=0, help="選び方をずらす（別の組み合わせを取る）")
    args = ap.parse_args(argv)

    zs = zipfile.ZipFile(HttpRangeFile(STEMS_URL))  # type: ignore[arg-type]
    zm = zipfile.ZipFile(HttpRangeFile(MIDI_URL))  # type: ignore[arg-type]
    wavs = [n for n in zs.namelist() if n.endswith(".wav")]
    # StemGMD/audio/<drummer>/eval_session/<kit>/<stem>/<track>.wav
    drummers = sorted({n.split("/")[2] for n in wavs})
    kits = sorted({n.split("/")[4] for n in wavs})
    tracks = sorted({n.split("/")[6] for n in wavs}, key=lambda t: int(t.split("_")[0]))
    midi_names = zm.namelist()
    out = Path(args.out)
    done = 0
    for i in range(args.n):
        j = i + args.offset
        kit = kits[j % len(kits)]
        track = tracks[(j + j // len(kits)) % len(tracks)]
        drummer = drummers[(j + j // len(kits) * 2) % len(drummers)]
        midi = next((m for m in midi_names if m.endswith(f"{drummer}/eval_session/{track[:-4]}.mid")), None)
        if midi is None:
            print(f"  MIDI が見つからない: {drummer} {track}", file=sys.stderr)
            continue
        groove = track.split("_")[1].split("-")[0]
        d = out / f"s{i + 1:02d}_{kit.replace('_', '')[:12]}_{groove}{track.split('_')[0]}"
        if (d / "drums.wav").exists() and (d / "hits.mid").exists():
            done += 1
            continue
        parts: dict[str, np.ndarray] = {}
        sr = 44100
        for stem, cls in STEM_TO_CLASS.items():
            x, sr = read_wav(zs, f"StemGMD/audio/{drummer}/eval_session/{kit}/{stem}/{track}")
            if args.seconds:
                x = x[: int(args.seconds * sr)]
            parts[cls] = parts[cls][: x.size] + x[: parts[cls].size] if cls in parts else x
        n = min(p.size for p in parts.values())
        mix = sum(p[:n] for p in parts.values())
        gain = 0.9 / (float(np.max(np.abs(mix))) + 1e-9)  # 全パートに同じ係数（和が drums.wav に等しいまま）
        d.mkdir(parents=True, exist_ok=True)
        sf.write(d / "drums.wav", (mix * gain).astype(np.float32), sr, subtype="PCM_16")
        for cls in CLASSES:
            sf.write(d / f"{cls}.wav", (parts[cls][:n] * gain).astype(np.float32), sr, subtype="PCM_16")
        (d / "hits.mid").write_bytes(zm.read(midi))
        (d / "start.txt").write_text("0", encoding="utf-8")  # 正解の MIDI と時刻を揃えるため、頭から評価する
        done += 1
        print(f"  {done:2d}/{args.n}  {kit:18s} {drummer}  {track}", file=sys.stderr)
    (out / "README.txt").write_text(
        "StemGMD (eval session) の一部。CC BY 4.0。https://zenodo.org/records/7882857\n"
        "Mezza, Giampiccolo, Zanoni, Bernardini, Sarti (2024) \"Toward Deep Drum Source Separation\", Pattern Recognition Letters.\n"
        "MIDI は Groove MIDI Dataset (Gillick et al. 2019), CC BY 4.0。analysis/fetch_stemgmd_sample.py で取得。\n"
        "パートは粗い 5 種にまとめ、モノラルにし、曲ごとに全パート共通の係数で音量をそろえてある。\n", encoding="utf-8")
    fp = zs.fp
    print(f"{done} 曲を {out} に置いた（要求 {fp.requests} 回、{fp.bytes_read / 1e6:.0f} MB）。", file=sys.stderr)
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
