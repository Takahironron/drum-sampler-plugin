#!/usr/bin/env python3
"""ドラムトラック（ch 10）のノート番号を、使うキットのパッド配置に合わせて付け替える。

生成した MIDI は GM ドラムマップだが、Maschine や XO のキットはパッドの並びが違う。
打点の時刻は変えないので、正解の `hits.csv` はそのまま使える。

使い方::

    python remap_midi_notes.py song.mid --list                      # 使っているノートと種類を表示
    python remap_midi_notes.py song.mid out.mid --map 36=36,38=37,42=39,46=40,49=44,51=45

`--map` は「元=先」をコンマ区切り。書いていないノートはそのまま。
よく使う配置は `--preset` で呼べる（配置はキットごとに違うので、鳴らして確かめてから使う）:
    maschine16  Maschine の 16 パッド（C1=36 から半音ずつ）に、キック→1、スネア→2、サイドスティック→3、
                閉ハット→4、ペダル→5、開ハット→6、タム低→7、タム中→8、タム高→9、クラッシュ→10、ライド→11、ベル→12 の順で置く
    xo8         XO の 8 パッド（C1=36 から半音ずつ）に、キック、スネア、閉ハット、開ハット、タム、クラッシュ、ライド、サイドスティック の順で置く
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drumeval.labels import GM_DRUM_MAP  # noqa: E402

PRESETS: dict[str, dict[int, int]] = {
    "maschine16": {36: 36, 38: 37, 37: 38, 42: 39, 44: 40, 46: 41, 41: 42, 45: 43, 48: 44, 49: 45, 51: 46, 53: 47},
    "xo8": {36: 36, 38: 37, 42: 38, 46: 39, 41: 40, 45: 40, 48: 40, 49: 41, 51: 42, 53: 42, 37: 43, 44: 38},
}


def parse_map(text: str) -> dict[int, int]:
    out: dict[int, int] = {}
    for pair in text.split(","):
        if not pair.strip():
            continue
        a, b = pair.split("=")
        out[int(a)] = int(b)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("midi")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--map", default=None)
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--list", action="store_true", help="ドラムで使っているノートを表示して終わる")
    ap.add_argument("--channel", type=int, default=9, help="ドラムのチャンネル（0 始まり。既定 9 = 10ch）")
    args = ap.parse_args(argv)

    import mido

    mid = mido.MidiFile(args.midi)
    used: dict[int, int] = {}
    for tr in mid.tracks:
        for msg in tr:
            if msg.type == "note_on" and msg.channel == args.channel and msg.velocity > 0:
                used[msg.note] = used.get(msg.note, 0) + 1
    if args.list or not args.out:
        for n in sorted(used):
            print(f"{n:3d}  {GM_DRUM_MAP.get(n, '?'):16s} {used[n]:5d} 打")
        if not args.out:
            return 0

    mapping = dict(PRESETS[args.preset]) if args.preset else {}
    if args.map:
        mapping.update(parse_map(args.map))
    if not mapping:
        print("--map か --preset が要る", file=sys.stderr)
        return 1
    changed = 0
    for tr in mid.tracks:
        for msg in tr:
            if msg.type in ("note_on", "note_off") and msg.channel == args.channel and msg.note in mapping:
                msg.note = mapping[msg.note]
                changed += 1
    mid.save(args.out)
    print(f"{changed} メッセージを付け替えて書き出した: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
