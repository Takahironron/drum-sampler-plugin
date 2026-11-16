#!/usr/bin/env python3
"""E-GMD（CC BY 4.0）から少数のトラックだけを取ってくる（zip 全体 90 GB を落とさない）。

HTTP の範囲指定（Range）で、zip の目録と、選んだトラックの音声・MIDI だけを読む。手元やクラウドで、本物のドラムの音と
正解の MIDI を使って解析（NMFD、切り出し、再現）を確かめるための小さな評価セットを作る::

    python analysis/fetch_egmd_sample.py --n 10 --out data/egmd_sample --stems-out data/egmd_stems
    python analysis/run_eval.py data/egmd_stems --parts nmfd-oracle --out data/results/egmd_sample.md

- `--out` には E-GMD と同じ配置（`e-gmd-v1.0.0.csv` と `drummerN/...`）で置く。`drumeval.egmd.read_index` でそのまま読める。
- `--stems-out` には `run_eval.py` の形（`<曲ID>/drums.wav` と正解の `hits.mid`）で置く。正解と音の時刻のずれ（E-GMD の配布物には
  ファイルごとに −73〜+119 ms のずれがある）を測って `label_offset.txt` に書く。`run_eval.py` はその値を正解の打点に足す。
- 選び方: 分割（既定 test）、`beat`（フィルでない）、長さ `--min-seconds`〜`--max-seconds`。キットが重ならないように 1 キット 1 本ずつ、
  スタイルも散るように選ぶ（乱数の種で決まる）。`--kits` でキット名の一部を指定できる。
- 出典の表示: Callender, Hawthorne, Engel (2020) "Improving Perceptual Quality of Drum Transcription with the Expanded Groove MIDI
  Dataset", arXiv:2004.00188。https://magenta.tensorflow.org/datasets/e-gmd 。ライセンスの整理は `docs/phase0/ライセンス一覧.md`。
- `data/` は `.gitignore` 済み。音声はコミットしない。
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0.zip"


class HttpRangeFile:
    """HTTP の範囲指定で読む、読み取り専用のファイルもどき（`zipfile.ZipFile` に渡せる）。ブロック単位で読んで少しだけ覚えておく。"""

    def __init__(self, url: str, block: int = 4 << 20, max_blocks: int = 24):
        self.url = url
        self.block = block
        self.max_blocks = max_blocks
        self.pos = 0
        self.blocks: dict[int, bytes] = {}
        self.requests = 0
        self.bytes_read = 0
        # 1 バイトだけ範囲指定で読み、206 と Content-Range が返るかで対応を確かめる（Accept-Ranges を返さないサーバーもある）
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            cr = r.headers.get("Content-Range", "")
            if r.status != 206 or "/" not in cr:
                raise OSError(f"範囲指定に対応していない: {url}")
            self.size = int(cr.rsplit("/", 1)[1])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else (self.pos + offset if whence == 1 else self.size + offset)
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        """[start, end) を読む。途中で切れたら続きから取り直す。"""
        out = bytearray()
        tries = 0
        while start + len(out) < end:
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={start + len(out)}-{end - 1}"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    out += r.read()
                self.requests += 1
            except OSError:
                tries += 1
                if tries > 4:
                    raise
        self.bytes_read += len(out)
        return bytes(out)

    def _block(self, i: int) -> bytes:
        if i not in self.blocks:
            if len(self.blocks) >= self.max_blocks:
                self.blocks.pop(next(iter(self.blocks)))
            self.blocks[i] = self._fetch(i * self.block, min((i + 1) * self.block, self.size))
        return self.blocks[i]

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(min(n, self.size - self.pos), 0)
        if n == 0:
            return b""
        if n > self.block * 4:  # 大きな読み込み（目録やファイル本体）は 1 回の要求で
            data = self._fetch(self.pos, self.pos + n)
        else:
            first, last = self.pos // self.block, (self.pos + n - 1) // self.block
            buf = b"".join(self._block(i) for i in range(first, last + 1))
            off = self.pos - first * self.block
            data = buf[off : off + n]
        self.pos += len(data)
        return data

    def close(self) -> None:
        self.blocks.clear()


def select_rows(rows: list[dict], n: int, split: str, min_sec: float, max_sec: float, kits: list[str], styles: list[str], seed: int) -> list[dict]:
    """1 キット 1 本ずつ、スタイルが散るように選ぶ。"""
    import random

    cand = []
    for r in rows:
        try:
            dur = float(r.get("duration") or 0.0)
        except ValueError:
            continue
        if split != "all" and r.get("split") != split:
            continue
        if r.get("beat_type") != "beat" or not (min_sec <= dur <= max_sec):
            continue
        if kits and not any(k.lower() in (r.get("kit_name") or "").lower() for k in kits):
            continue
        if styles and not any(s.lower() in (r.get("style") or "").lower() for s in styles):
            continue
        cand.append(r)
    rng = random.Random(seed)
    rng.shuffle(cand)
    by_kit: dict[str, list[dict]] = {}
    for r in cand:
        by_kit.setdefault(r.get("kit_name") or "", []).append(r)
    kit_names = sorted(by_kit)
    rng.shuffle(kit_names)
    picked: list[dict] = []
    used_styles: dict[str, int] = {}
    used_perf: set[str] = set()  # 同じ演奏（MIDI）を別のキットで重ねて選ばない
    while len(picked) < n and kit_names:
        progressed = False
        for kit in list(kit_names):
            if len(picked) >= n:
                break
            pool = [r for r in by_kit[kit] if r not in picked and (r.get("midi_filename") or "").rsplit("_", 1)[0] not in used_perf]
            if not pool:
                kit_names.remove(kit)
                continue
            pool.sort(key=lambda r: used_styles.get((r.get("style") or "").split("/")[0], 0))  # まだ少ないスタイルを先に
            r = pool[0]
            picked.append(r)
            used_perf.add((r.get("midi_filename") or "").rsplit("_", 1)[0])
            st = (r.get("style") or "").split("/")[0]
            used_styles[st] = used_styles.get(st, 0) + 1
            progressed = True
        if not progressed:
            break
    return picked


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())[:14] or "x"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--out", default=str(root / "data" / "egmd_sample"), help="E-GMD と同じ配置で置くフォルダ")
    ap.add_argument("--stems-out", default=None, help="run_eval.py の形（<曲ID>/drums.wav と hits.mid）でも置くフォルダ")
    ap.add_argument("--n", type=int, default=10, help="取ってくるトラックの数")
    ap.add_argument("--split", default="test", help="train / validation / test / all")
    ap.add_argument("--min-seconds", type=float, default=25.0)
    ap.add_argument("--max-seconds", type=float, default=90.0)
    ap.add_argument("--kits", default="", help="キット名の一部をコンマ区切りで（例: Acoustic,Jazz,808）")
    ap.add_argument("--styles", default="", help="スタイル名の一部をコンマ区切りで（例: rock,funk,jazz）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-mb", type=float, default=80.0, help="1 ファイルの大きさの上限（MB）。超えるトラックは飛ばす")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--list-kits", action="store_true", help="キット名と本数を出して終わる（取ってこない）")
    args = ap.parse_args(argv)

    fp = HttpRangeFile(args.url)
    print(f"zip {fp.size / 1e9:.1f} GB。目録を読む...", file=sys.stderr)
    with zipfile.ZipFile(fp) as z:  # type: ignore[arg-type]
        names = z.namelist()
        csv_name = next(n for n in names if n.endswith(".csv"))
        prefix = csv_name.rsplit("/", 1)[0] + "/" if "/" in csv_name else ""
        with z.open(csv_name) as f:
            rows = list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")))
        if args.list_kits:
            from collections import Counter

            c = Counter((r.get("kit_name"), r.get("split")) for r in rows)
            for kit in sorted({k for k, _ in c}):
                print(f"{kit}: " + " ".join(f"{s} {c[(kit, s)]}" for s in ("train", "validation", "test") if c.get((kit, s))))
            return 0
        picked = select_rows(rows, args.n * 2, args.split, args.min_seconds, args.max_seconds,
                             [k for k in args.kits.split(",") if k.strip()], [s for s in args.styles.split(",") if s.strip()], args.seed)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        done: list[dict] = []
        for r in picked:
            if len(done) >= args.n:
                break
            members = [prefix + r["audio_filename"], prefix + r["midi_filename"]]
            try:
                infos = [z.getinfo(m) for m in members]
            except KeyError:
                continue
            if any(i.file_size > args.max_mb * 1e6 for i in infos):
                continue
            for m, key in zip(members, ("audio_filename", "midi_filename")):
                dst = out / r[key]
                if dst.exists() and dst.stat().st_size == z.getinfo(m).file_size:
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as src, open(dst, "wb") as dstf:
                    shutil.copyfileobj(src, dstf, 1 << 20)
            done.append(r)
            print(f"  {len(done):2d}/{args.n}  {r['kit_name']:28s} {r['style']:22s} {float(r['duration']):5.1f} 秒  {r['audio_filename']}", file=sys.stderr)
    with open(out / "e-gmd-v1.0.0.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(done)
    (out / "README.txt").write_text(
        "E-GMD (Expanded Groove MIDI Dataset) の一部。CC BY 4.0。\n"
        "Callender, Hawthorne, Engel (2020), arXiv:2004.00188. https://magenta.tensorflow.org/datasets/e-gmd\n"
        "analysis/fetch_egmd_sample.py で取得。\n", encoding="utf-8")
    if args.stems_out:
        stems = Path(args.stems_out)
        for i, r in enumerate(done, 1):
            d = stems / f"e{i:02d}_{slug(r['kit_name'])}_{slug(r['style'].split('/')[0])}"
            d.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out / r["audio_filename"], d / "drums.wav")
            shutil.copyfile(out / r["midi_filename"], d / "hits.mid")
            (d / "start.txt").write_text("0", encoding="utf-8")  # 正解の MIDI と時刻を揃えるため、頭から評価する
        # E-GMD の正解の MIDI は、ファイルごとに音とずれていることがある。測って label_offset.txt に書く（run_eval.py が使う）
        from egmd_offsets import run_stems

        run_stems(stems)
    print(f"{len(done)} 本を {out} に置いた（要求 {fp.requests} 回、{fp.bytes_read / 1e6:.0f} MB）。" + (f" run_eval 用: {args.stems_out}" if args.stems_out else ""), file=sys.stderr)
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
