"""打点ラベルの読み書き。

ラベルの形式は2つ。

1. CSV: 1行が1打。列は `time_sec,class[,velocity]`。ヘッダ行は任意。
   例::

       time_sec,class,velocity
       0.000,kick,110
       0.250,hihat_closed,70

2. MIDI: General MIDI のドラムマップ（チャンネル10）を前提に、ノートオンを打点として読む。
   自分の曲の MIDI から書き出したステムの正解に使う。

種類（class）は細かい名前（`hihat_closed` など）で持ち、採点のときに
粗い5種（kick / snare / hihat / tom / cymbal）にまとめる（計画書 3.2 の Phase 1 の粒度）。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# 細かい種類 → 粗い5種
FINE_TO_COARSE: dict[str, str] = {
    "kick": "kick",
    "snare": "snare",
    "snare_rim": "snare",
    "snare_sidestick": "snare",
    "hihat": "hihat",
    "hihat_closed": "hihat",
    "hihat_open": "hihat",
    "hihat_pedal": "hihat",
    "tom": "tom",
    "tom_high": "tom",
    "tom_mid": "tom",
    "tom_low": "tom",
    "tom_floor": "tom",
    "cymbal": "cymbal",
    "ride": "cymbal",
    "ride_bell": "cymbal",
    "crash": "cymbal",
    "china": "cymbal",
    "splash": "cymbal",
}

COARSE_CLASSES: tuple[str, ...] = ("kick", "snare", "hihat", "tom", "cymbal")

# M0-1 の判定に使う3種
M0_CLASSES: tuple[str, ...] = ("kick", "snare", "hihat")

# General MIDI のドラムマップ（ノート番号 → 細かい種類）。載っていない番号は無視する。
# Roland TD-17（E-GMD の録音に使われた電子ドラム）の追加ノートも含む: 22 閉ハット縁、26 開ハット縁、58 フロアタム縁。
GM_DRUM_MAP: dict[int, str] = {
    22: "hihat_closed",
    26: "hihat_open",
    35: "kick",
    36: "kick",
    37: "snare_sidestick",
    38: "snare",
    39: "snare",  # hand clap。粗い種類ではスネアに寄せる（要検討）
    40: "snare",
    41: "tom_floor",
    42: "hihat_closed",
    43: "tom_floor",
    44: "hihat_pedal",
    45: "tom_low",
    46: "hihat_open",
    47: "tom_mid",
    48: "tom_mid",
    49: "crash",
    50: "tom_high",
    51: "ride",
    52: "china",
    53: "ride_bell",
    55: "splash",
    57: "crash",
    58: "tom_floor",
    59: "ride",
}


@dataclass(frozen=True)
class Hit:
    """1打。`time` は秒、`cls` は細かい種類、`velocity` は 1〜127（不明なら None）。"""

    time: float
    cls: str
    velocity: int | None = None


def to_coarse(cls: str) -> str:
    """細かい種類を粗い5種に直す。知らない名前はそのまま返す。"""
    return FINE_TO_COARSE.get(cls, cls)


def load_hits(path: str | Path) -> list[Hit]:
    """拡張子で CSV か MIDI かを判断して読む。"""
    p = Path(path)
    if p.suffix.lower() in (".mid", ".midi"):
        return load_hits_midi(p)
    return load_hits_csv(p)


def load_hits_csv(path: str | Path) -> list[Hit]:
    hits: list[Hit] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            # ヘッダ行（time_sec など）は読み飛ばす
            try:
                t = float(row[0])
            except ValueError:
                continue
            cls = row[1].strip()
            vel = None
            if len(row) > 2 and row[2].strip():
                vel = int(float(row[2]))
            hits.append(Hit(t, cls, vel))
    hits.sort(key=lambda h: h.time)
    return hits


def save_hits_csv(hits: Iterable[Hit], path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_sec", "class", "velocity"])
        for h in sorted(hits, key=lambda h: h.time):
            w.writerow([f"{h.time:.6f}", h.cls, "" if h.velocity is None else h.velocity])


def load_hits_midi(path: str | Path, note_map: dict[int, str] | None = None, channel: int | None = 9) -> list[Hit]:
    """MIDI ファイルのノートオン（velocity > 0）を打点として読む。

    すべてのトラックをまとめ、テンポ変化も `mido` に任せて秒に直す。
    `channel` はドラムのチャンネル（0 始まりで 9 = GM の 10ch）。そのチャンネルにノートが無ければ全チャンネルを読む
    （ドラムだけを別チャンネルで書き出したファイル向け）。`channel=None` で常に全チャンネル。
    """
    import mido  # 遅延 import（CSV だけのときは要らない）

    note_map = GM_DRUM_MAP if note_map is None else note_map
    mid = mido.MidiFile(str(path))
    events: list[tuple[float, int, int, int]] = []  # (秒, channel, note, velocity)
    t = 0.0
    for msg in mid:  # mido が各メッセージの時間差を秒で返す
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            events.append((t, msg.channel, msg.note, msg.velocity))
    if channel is not None and any(ch == channel for _, ch, _, _ in events):
        events = [e for e in events if e[1] == channel]
    hits = [Hit(t, note_map[n], v) for t, _, n, v in events if n in note_map]
    hits.sort(key=lambda h: h.time)
    return hits
