#!/usr/bin/env python3
"""song.mid を合成音で WAV にする（確認用の仮の音。DAW の VST で書き出す前にパターンと構成を聞くため）。

使い方::

    python render_midi_synth.py analysis/eval_midi/b01_rock/song.mid 出力.wav [--sr 22050] [--drums-only]

- ドラム（ch 10）は `drumeval.synth` のキック・スネア・ハイハット・タム・シンバル。
- ベース（ch 1）はサイン＋倍音、コード（ch 2）は柔らかいノコギリ波、リード（ch 3）は矩形波。どれも短いエンベロープだけ。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drumeval.labels import GM_DRUM_MAP, Hit  # noqa: E402
from drumeval.synth import render as render_drums  # noqa: E402


def read_notes(path: Path):
    import mido

    mid = mido.MidiFile(str(path))
    t = 0.0
    on: dict[tuple[int, int], tuple[float, int]] = {}
    notes: list[tuple[float, float, int, int, int]] = []  # (start, end, channel, note, velocity)
    for msg in mid:
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            if msg.channel == 9:  # ドラムは打点だけでよい（同じ音の連打で対応付けを落とさない）
                notes.append((t, t + 0.1, 9, msg.note, msg.velocity))
                continue
            on[(msg.channel, msg.note)] = (t, msg.velocity)
        elif msg.type in ("note_off", "note_on"):
            key = (msg.channel, msg.note)
            if key in on:
                s, v = on.pop(key)
                notes.append((s, t, msg.channel, msg.note, v))
    return notes, t


def _env(n: int, sr: int, attack: float, release: float) -> np.ndarray:
    e = np.ones(n)
    a = min(int(attack * sr), n)
    r = min(int(release * sr), n)
    if a > 0:
        e[:a] = np.linspace(0, 1, a)
    if r > 0:
        e[-r:] *= np.linspace(1, 0, r)
    return e


def render_melodic(notes, duration: float, sr: int) -> np.ndarray:
    out = np.zeros(int(duration * sr) + sr)
    for start, end, ch, note, vel in notes:
        if ch == 9:
            continue
        f = 440.0 * 2 ** ((note - 69) / 12)
        n = max(int((end - start) * sr), int(0.05 * sr))
        t = np.arange(n) / sr
        g = vel / 127.0
        if ch == 0:  # ベース
            y = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t)) * np.exp(-t * 3.0) * 0.5 * g
            y *= _env(n, sr, 0.003, 0.02)
        elif ch == 1:  # コード: 柔らかいノコギリ波（倍音を 6 つ）
            y = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 7)) * 0.08 * g
            y *= _env(n, sr, 0.05, 0.1)
        else:  # リード: 矩形波（奇数倍音）
            y = sum(np.sin(2 * np.pi * f * k * t) / k for k in (1, 3, 5, 7)) * 0.12 * g
            y *= _env(n, sr, 0.005, 0.03)
        s = int(start * sr)
        out[s : s + n] += y[: out.size - s]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("midi")
    ap.add_argument("out_wav")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--drums-only", action="store_true")
    args = ap.parse_args(argv)

    import soundfile as sf

    notes, length = read_notes(Path(args.midi))
    duration = length + 1.5
    hits = [Hit(s, GM_DRUM_MAP[n], v) for s, e, ch, n, v in notes if ch == 9 and n in GM_DRUM_MAP]
    drums = render_drums(hits, sr=args.sr, duration=duration)
    mix = drums if args.drums_only else drums * 0.8 + render_melodic(notes, duration, args.sr)[: drums.size]
    mix = mix / (np.max(np.abs(mix)) + 1e-9) * 0.9
    sf.write(args.out_wav, mix.astype(np.float32), args.sr, subtype="PCM_16")
    print(f"{args.out_wav}: {duration:.1f} 秒, ドラム {len(hits)} 打, 音符 {len(notes) - len(hits)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
