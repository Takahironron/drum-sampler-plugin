#!/usr/bin/env python3
"""iTunes／Apple Music の「ライブラリの書き出し」（XML）か、プレイリストのテキスト書き出しを読み、
区分 A の候補を選ぶための一覧（ジャンル別、年順）を Markdown で出す。

使い方::

    python list_itunes_library.py ライブラリ.xml [--out docs/phase0/itunes一覧.md] [--min-seconds 90]
    python list_itunes_library.py プレイリスト.txt --out ...

- XML: iTunes／Music の「ファイル → ライブラリ → ライブラリを書き出し…」で作る plist。
- テキスト: 曲を選んで「ファイル → ライブラリ → プレイリストを書き出し…」でタブ区切りのテキスト。
- 出すのは アーティスト・曲名・アルバム・年・ジャンル・長さ だけ。音声ファイルには触れない。
"""

from __future__ import annotations

import argparse
import plistlib
import sys
from collections import defaultdict
from pathlib import Path


def read_xml(path: Path) -> list[dict]:
    with open(path, "rb") as f:
        lib = plistlib.load(f)
    out = []
    for t in lib.get("Tracks", {}).values():
        if t.get("Kind", "").lower().find("audio") < 0 and "Total Time" not in t:
            continue
        out.append({
            "artist": t.get("Artist", ""), "name": t.get("Name", ""), "album": t.get("Album", ""),
            "year": t.get("Year", ""), "genre": t.get("Genre", "（なし）"), "seconds": int(t.get("Total Time", 0)) // 1000,
        })
    return out


def read_txt(path: Path) -> list[dict]:
    raw = path.read_bytes()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", errors="replace")
    lines = [l for l in text.splitlines() if l.strip()]
    header = lines[0].split("\t")
    idx = {h: i for i, h in enumerate(header)}

    def col(row, *names):
        for n in names:
            if n in idx and idx[n] < len(row):
                return row[idx[n]]
        return ""

    out = []
    for l in lines[1:]:
        row = l.split("\t")
        secs = col(row, "時間", "Time")
        try:
            m, s = secs.split(":")
            seconds = int(m) * 60 + int(s)
        except ValueError:
            seconds = 0
        out.append({"artist": col(row, "アーティスト", "Artist"), "name": col(row, "名前", "Name"), "album": col(row, "アルバム", "Album"),
                    "year": col(row, "年", "Year"), "genre": col(row, "ジャンル", "Genre") or "（なし）", "seconds": seconds})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-seconds", type=int, default=90)
    args = ap.parse_args(argv)
    p = Path(args.path)
    tracks = read_xml(p) if p.suffix.lower() in (".xml", ".plist") else read_txt(p)
    tracks = [t for t in tracks if t["seconds"] >= args.min_seconds]
    by_genre: dict[str, list[dict]] = defaultdict(list)
    for t in tracks:
        by_genre[t["genre"]].append(t)
    lines = [f"# iTunes ライブラリの一覧（{len(tracks)} 曲、{len(by_genre)} ジャンル）", ""]
    for g in sorted(by_genre, key=lambda g: -len(by_genre[g])):
        lines += [f"## {g}（{len(by_genre[g])}）", "", "| アーティスト | 曲 | アルバム | 年 | 長さ |", "|---|---|---|---|---|"]
        for t in sorted(by_genre[g], key=lambda t: (str(t["year"]), t["artist"], t["name"])):
            lines.append(f"| {t['artist']} | {t['name']} | {t['album']} | {t['year']} | {t['seconds'] // 60}:{t['seconds'] % 60:02d} |")
        lines.append("")
    md = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print("書き出した:", args.out, file=sys.stderr)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
