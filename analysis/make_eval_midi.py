#!/usr/bin/env python3
"""区分 B の評価用「曲」を MIDI で自動生成する（正解の打点と区間ラベル付き）。

使い方::

    python make_eval_midi.py [--out data/stems] [--seed 0]

7 曲（ジャンルとテンポを散らす）を `data/stems/b01_rock/` ... に書く。各フォルダ:

    song.mid       ドラム（ch 10、GM ドラムマップ）＋ベース（ch 1）＋コード（ch 2）＋リード（ch 3）の MIDI
    hits.csv       ドラムの打点の正解（song.mid から読み直したもの。time_sec,class,velocity）
    sections.csv   区間のラベル（start_sec,end_sec,label）
    README.md      書き出しの手順

手順（Takahiro の DAW で）:
    1. song.mid を DAW に読み込み、トラックごとに手持ちの VST を割り当てる（ドラムは Maschine などの生ドラム系キット）。
    2. 曲の頭（1 小節目の頭 = 0 秒）から 2mix を WAV で書き出す（先頭に無音を足さない。足したら hits.csv の時刻がずれる）。
    3. SpectraLayers（や Demucs）でドラムを分離し、`drums.wav` としてこのフォルダに置く。
    4. `python analysis/run_eval.py data/stems` で F 値が出る。

打点はドラムの MIDI そのものなので、書き出しの先頭が 0 秒であれば正解は正確。音源のアタックの遅れ（数 ms）は許容誤差（±50 ms）の中。
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drumeval.labels import load_hits_midi, save_hits_csv  # noqa: E402

TPB = 480  # ticks per beat

# GM ドラムマップ
KICK, SIDESTICK, SNARE, HH_CLOSED, HH_PEDAL, HH_OPEN = 36, 37, 38, 42, 44, 46
TOM_LOW, TOM_MID, TOM_HIGH, CRASH, RIDE, RIDE_BELL = 41, 45, 48, 49, 51, 53


@dataclass
class Style:
    id: str
    name: str
    bpm: float
    swing: float  # 0〜0.33: 16分の裏をどれだけ遅らせるか
    hat: int  # 基本のハイハット（閉）か ride
    hat_grid: int  # 8 → 8分、16 → 16分
    kick_pattern: list[float]  # 拍の位置（0〜4）
    snare_pattern: list[float]
    ghost_prob: float
    open_hat_prob: float  # 裏拍でハイハットを開ける確率
    double_kick: bool
    vel_range: tuple[int, int]
    bass_root: int  # ベースのルート（MIDI ノート）
    chords: list[list[int]]  # 1 小節ごとのコード（4 小節で回す）
    timing_sd_ms: float
    sections: list[tuple[str, int]] = field(default_factory=lambda: [("Intro", 4), ("Verse", 8), ("Chorus", 8), ("Verse", 8), ("Chorus", 8), ("Outro", 4)])


def _maj(root):
    return [root, root + 4, root + 7]


def _min(root):
    return [root, root + 3, root + 7]


STYLES: list[Style] = [
    Style("b01_rock", "ロック（8ビート）", 122, 0.0, HH_CLOSED, 8, [0, 2, 2.5], [1, 3], 0.05, 0.15, False, (95, 127), 40,
          [_maj(52), _maj(57), _min(50), _maj(55)], 6),
    Style("b02_pop", "ポップ（16ビート、軽い）", 100, 0.05, HH_CLOSED, 16, [0, 1.75, 2.5], [1, 3], 0.2, 0.1, False, (70, 115), 41,
          [_maj(53), _min(57), _min(50), _maj(58)], 5),
    Style("b03_hiphop", "ヒップホップ（スウィング、遅い）", 88, 0.2, HH_CLOSED, 8, [0, 1.5, 2.75], [1, 3], 0.1, 0.05, False, (90, 127), 38,
          [_min(50), _min(50), _maj(46), _maj(48)], 4),
    Style("b04_funk", "ファンク（16分、ゴースト多め）", 108, 0.08, HH_CLOSED, 16, [0, 0.75, 2.5], [1, 3, 3.75], 0.45, 0.25, False, (60, 120), 43,
          [_min(55), _min(55), _maj(60), _min(55)], 4),
    Style("b05_metal", "メタル（速い、ツーバス）", 176, 0.0, RIDE, 8, [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5], [1, 3], 0.0, 0.0, True, (100, 127), 40,
          [_min(52), _min(52), _maj(48), _min(50)], 3),
    Style("b06_ballad", "バラード（遅い、ライド）", 72, 0.0, RIDE, 8, [0, 2.5], [1, 3], 0.1, 0.0, False, (55, 105), 43,
          [_maj(55), _min(59), _maj(60), _maj(62)], 8),
    Style("b07_electronic", "エレクトロ（4つ打ち、機械的）", 128, 0.0, HH_CLOSED, 16, [0, 1, 2, 3], [1, 3], 0.0, 0.5, False, (110, 127), 45,
          [_min(57), _min(57), _maj(53), _maj(55)], 1),
]


def build_song(style: Style, rng: np.random.Generator):
    import mido

    mid = mido.MidiFile(type=1, ticks_per_beat=TPB)
    beat_sec = 60.0 / style.bpm

    def sec(beats: float) -> float:
        return beats * beat_sec

    # 区間
    sections: list[tuple[float, float, str]] = []
    bar = 0
    bar_labels: list[str] = []
    for label, n_bars in style.sections:
        sections.append((sec(bar * 4), sec((bar + n_bars) * 4), label))
        bar_labels += [label] * n_bars
        bar += n_bars
    n_bars = bar

    drum_events: list[tuple[int, int, int]] = []  # (tick, note, velocity)
    bass_events: list[tuple[int, int, int, int]] = []  # (tick, note, vel, length)
    chord_events: list[tuple[int, list[int], int, int]] = []
    lead_events: list[tuple[int, int, int, int]] = []

    def ticks(beats: float) -> int:
        return int(round(beats * TPB))

    def humanize(tick: int) -> int:
        return max(0, tick + int(round(rng.normal(0, style.timing_sd_ms) / 1000.0 / beat_sec * TPB)))

    def vel(lo_hi: tuple[int, int], scale: float = 1.0) -> int:
        lo, hi = lo_hi
        return int(np.clip(rng.integers(lo, hi + 1) * scale, 1, 127))

    for b in range(n_bars):
        label = bar_labels[b]
        base = b * 4
        is_last_of_section = (b + 1 < n_bars and bar_labels[b + 1] != label) or b == n_bars - 1
        is_first_of_section = b == 0 or bar_labels[b - 1] != label
        density = {"Intro": 0.6, "Verse": 0.85, "Chorus": 1.0, "Outro": 0.6}[label]
        loud = {"Intro": 0.8, "Verse": 0.9, "Chorus": 1.0, "Outro": 0.75}[label]

        # ハイハット／ライド
        grid = style.hat_grid
        for i in range(grid):
            pos = i * 4 / grid
            off = pos % 1 != 0 and (pos * 4) % 2 == 1  # 16分の裏
            if label == "Intro" and grid == 16 and i % 2 == 1:
                continue
            note = style.hat
            if style.hat == HH_CLOSED and off is False and (pos % 1) == 0.5 and rng.random() < style.open_hat_prob:
                note = HH_OPEN
            if style.hat == RIDE and label == "Chorus" and pos % 1 == 0 and rng.random() < 0.3:
                note = RIDE_BELL
            swing_shift = style.swing * (4 / grid) if (grid == 16 and i % 2 == 1) or (grid == 8 and i % 2 == 1 and style.swing > 0.1) else 0.0
            accent = 1.0 if pos % 1 == 0 else 0.7
            drum_events.append((humanize(ticks(base + pos + swing_shift)), note, vel(style.vel_range, accent * loud * 0.85)))

        # キック
        kicks = style.kick_pattern if (not style.double_kick or label == "Chorus") else style.kick_pattern[::2]
        for pos in kicks:
            if rng.random() > density + 0.1:
                continue
            drum_events.append((humanize(ticks(base + pos)), KICK, vel(style.vel_range, loud)))

        # スネア（Intro は sidestick）
        for pos in style.snare_pattern:
            note = SIDESTICK if label == "Intro" else SNARE
            drum_events.append((humanize(ticks(base + pos)), note, vel(style.vel_range, loud)))
        # ゴースト
        for i in range(16):
            pos = i / 4
            if pos in style.snare_pattern or pos in kicks:
                continue
            if rng.random() < style.ghost_prob * density:
                drum_events.append((humanize(ticks(base + pos + style.swing / 4 * (i % 2))), SNARE, vel((25, 50))))

        # 区間の頭にクラッシュ、区間の終わりにフィル
        if is_first_of_section and label in ("Chorus", "Verse") and b > 0:
            drum_events.append((ticks(base), CRASH, vel((100, 127))))
        if is_last_of_section and label != "Outro":
            toms = [SNARE, SNARE, TOM_HIGH, TOM_HIGH, TOM_MID, TOM_MID, TOM_LOW, TOM_LOW]
            start = 2.0 if style.bpm < 150 else 3.0
            n = 8 if style.bpm < 150 else 4
            for i in range(n):
                pos = start + i * (4 - start) / n
                drum_events.append((humanize(ticks(base + pos)), toms[i * 8 // n], vel((85, 120))))
        if b == n_bars - 1:
            drum_events.append((ticks(base + 4), CRASH, 120))
            drum_events.append((ticks(base + 4), KICK, 120))

        # ベース（キックに合わせる＋8分）
        chord = style.chords[b % 4]
        root = style.bass_root + (chord[0] - style.chords[0][0])
        for pos in sorted(set([0, 2] + [p for p in kicks if p < 4])):
            bass_events.append((ticks(base + pos), root, vel((80, 110), loud), ticks(0.45)))
        if label == "Chorus":
            for pos in (1.5, 3.5):
                bass_events.append((ticks(base + pos), root + 7 if rng.random() < 0.5 else root, vel((70, 95)), ticks(0.4)))

        # コード（全音符、Intro と Outro は薄く）
        chord_events.append((ticks(base), [n + 12 for n in chord], vel((60, 85), 0.8 if label in ("Intro", "Outro") else 1.0), ticks(3.8)))

        # リード（Chorus だけ、4分と8分）
        if label == "Chorus":
            scale = [chord[0] + 24, chord[1] + 24, chord[2] + 24, chord[0] + 31]
            for pos in (0, 1, 1.5, 2, 3):
                lead_events.append((ticks(base + pos), int(rng.choice(scale)), vel((75, 100)), ticks(0.45)))

    def track(name: str, channel: int, events: list[tuple[int, int, int, int]], program: int | None = None) -> "mido.MidiTrack":
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage("track_name", name=name, time=0))
        if program is not None:
            tr.append(mido.Message("program_change", channel=channel, program=program, time=0))
        msgs: list[tuple[int, int, mido.Message]] = []
        for tick, note, v, length in events:
            msgs.append((tick, 1, mido.Message("note_on", channel=channel, note=note, velocity=v, time=0)))
            msgs.append((tick + length, 0, mido.Message("note_off", channel=channel, note=note, velocity=0, time=0)))
        msgs.sort(key=lambda m: (m[0], m[1]))
        last = 0
        for tick, _, msg in msgs:
            msg.time = tick - last
            last = tick
            tr.append(msg)
        tr.append(mido.MetaMessage("end_of_track", time=0))
        return tr

    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name=style.id, time=0))  # MIDI のメタ文字列は latin-1 なので日本語は入れない
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(style.bpm), time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    # 区間をマーカーに
    last = 0
    for start, _, label in sections:
        tick = int(round(start / beat_sec * TPB))
        meta.append(mido.MetaMessage("marker", text=label, time=tick - last))
        last = tick
    meta.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(meta)
    mid.tracks.append(track("Drums", 9, [(t, n, v, ticks(0.1)) for t, n, v in drum_events]))
    mid.tracks.append(track("Bass", 0, bass_events, program=33))
    mid.tracks.append(track("Chords", 1, [(t, n, v, ln) for t, notes, v, ln in chord_events for n in notes], program=4))
    mid.tracks.append(track("Lead", 2, lead_events, program=80))
    return mid, sections


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "stems"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out_root = Path(args.out)
    for k, style in enumerate(STYLES):
        rng = np.random.default_rng(args.seed * 100 + k)
        mid, sections = build_song(style, rng)
        d = out_root / style.id
        d.mkdir(parents=True, exist_ok=True)
        mid.save(str(d / "song.mid"))
        hits = load_hits_midi(d / "song.mid")
        save_hits_csv(hits, d / "hits.csv")
        with open(d / "sections.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["start_sec", "end_sec", "label"])
            for s, e, label in sections:
                w.writerow([f"{s:.3f}", f"{e:.3f}", label])
        (d / "README.md").write_text(
            f"# {style.id}: {style.name}（{style.bpm:g} BPM、{sections[-1][1]:.0f} 秒、打点 {len(hits)}）\n\n"
            "1. `song.mid` を DAW に読み込み、Drums（ch 10）に生ドラム系の VST、Bass／Chords／Lead に任意の VST を割り当てる。\n"
            "2. 1 小節目の頭（0 秒）から 2mix を WAV で書き出す（先頭に無音を足さない）。\n"
            "3. SpectraLayers（や Demucs）でドラムを分離し、`drums.wav` としてこのフォルダに置く。\n"
            "4. `python analysis/run_eval.py data/stems` で F 値を出す。\n\n"
            "`hits.csv` は `song.mid` のドラムの打点（正解）、`sections.csv` は区間のラベル。\n",
            encoding="utf-8",
        )
        print(f"{style.id:16s} {style.name:20s} {style.bpm:5.0f} BPM  {sections[-1][1]:5.1f} 秒  打点 {len(hits):4d}  → {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
