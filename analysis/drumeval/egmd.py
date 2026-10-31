"""E-GMD（Expanded Groove MIDI Dataset、CC BY 4.0）で採譜モデルを学習するための読み込みと増強。

- 一覧: ルートの `e-gmd-v1.0.0.csv`
  （drummer, session, id, style, bpm, beat_type, time_signature, midi_filename, audio_filename, duration, split, kit_name）。
- 音声: 44.1 kHz、24 bit。モデルは 22.05 kHz を使うので 1/2 に間引く。
- MIDI: Roland TD-17 のノート配置。GM に近いが縁（リム）や開閉の別ノートがあり、`labels.GM_DRUM_MAP` に足してある。
- 音声と MIDI は 2 ms 以内に揃っている（データセットの説明。https://magenta.tensorflow.org/datasets/e-gmd）。
- 出典の表示: Callender, Hawthorne, Engel (2020) "Improving Perceptual Quality of Drum Transcription with the Expanded Groove MIDI Dataset", arXiv:2004.00188。

増強（`EGMDChunks`）: 音量、EQ（低域／高域の落とし）、**にじみ**（ドラム以外のステムを混ぜる。分離ツールの残りを模す）、
雑音、飽和、短い残響。にじみに CD 曲のステムを使う場合は評価用のモデルに限る（製品のモデルには権利の明確な音だけを使う。R3）。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, lfilter, resample_poly

from .features import FeatureConfig, frame_targets, log_mel
from .labels import Hit, load_hits_midi

CLASSES5: tuple[str, ...] = ("kick", "snare", "hihat", "tom", "cymbal")
AUDIO_EXT = (".wav", ".flac", ".aif", ".aiff", ".ogg")


@dataclass
class Track:
    audio: Path
    midi: Path
    split: str
    duration: float
    kit: str
    style: str


def read_index(root: str | Path) -> list[Track]:
    """`e-gmd-v1.0.0.csv` を読み、音声と MIDI の両方があるものだけ返す。"""
    root = Path(root)
    csvs = sorted(root.glob("*.csv")) or sorted(root.glob("*/*.csv"))
    if not csvs:
        raise FileNotFoundError(f"E-GMD の CSV が見つからない: {root}")
    csv_path = csvs[0]
    base = csv_path.parent
    tracks: list[Track] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a, m = row.get("audio_filename") or "", row.get("midi_filename") or ""
            if not a or not m:
                continue
            ap, mp = base / a, base / m
            if not ap.exists() or not mp.exists():
                continue
            try:
                dur = float(row.get("duration") or 0.0)
            except ValueError:
                dur = 0.0
            tracks.append(Track(ap, mp, row.get("split", "train") or "train", dur, row.get("kit_name", ""), row.get("style", "")))
    return tracks


def load_audio(path: str | Path, sr_target: int, start_sec: float = 0.0, dur_sec: float | None = None) -> np.ndarray:
    """ファイルの一部だけを読み、モノラルにして `sr_target` に間引く。"""
    info = sf.info(str(path))
    sr = info.samplerate
    start = int(start_sec * sr)
    frames = -1 if dur_sec is None else int(dur_sec * sr)
    x, _ = sf.read(str(path), start=start, frames=frames, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != sr_target:
        g = gcd(sr, sr_target)
        x = resample_poly(x, sr_target // g, sr // g).astype(np.float32)
    return x


def collect_bleed_files(folder: str | Path) -> list[Path]:
    """にじみ用の音声（ドラム以外のステム）。ドラム系の名前と、評価で作った WAV は除く。"""
    skip = ("drum", "ドラム", "キック", "スネア", "ハイハット", "シンバル", "タム", "kick", "snare", "hihat", "cymbal", "tom",
            "rerender", "original_cut", "audition")
    out = []
    for p in sorted(Path(folder).rglob("*")):
        if p.suffix.lower() in AUDIO_EXT and not any(k in p.stem.lower() for k in skip):
            out.append(p)
    return out


def estimate_offset(x: np.ndarray, sr: int, hits: list[Hit], cfg: FeatureConfig, max_lag_sec: float = 0.1) -> float:
    """音声のエネルギーの立ち上がりと MIDI の打点の相互相関から、音声の遅れ（秒。正なら音声が遅い）を見積もる。確認用。"""
    from .baseline import BaselineConfig, spectrogram

    bcfg = BaselineConfig(n_fft=cfg.n_fft, hop=cfg.hop)
    _, t, mag = spectrogram(x, sr, bcfg)
    e = 10.0 * np.log10(np.sum(mag**2, axis=0) + 1e-8)
    rise = np.maximum(e - np.concatenate([e[:1], e[:-1]]), 0.0)
    train = np.zeros_like(rise)
    for h in hits:
        i = int(round(h.time * sr / cfg.hop))
        if 0 <= i < train.size:
            train[i] = 1.0
    max_lag = int(max_lag_sec * sr / cfg.hop)
    best, best_lag = -1.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            c = float(np.dot(rise[lag:], train[: train.size - lag]))
        else:
            c = float(np.dot(rise[: rise.size + lag], train[-lag:]))
        if c > best:
            best, best_lag = c, lag
    return best_lag * cfg.hop / sr


def augment(x: np.ndarray, sr: int, rng: np.random.Generator, bleed_files: list[Path] | None = None) -> np.ndarray:
    """学習用の増強。入力はモノラル float32。"""
    y = x.astype(np.float64)
    # 音量
    y *= 10 ** (rng.uniform(-12.0, 6.0) / 20)
    # EQ: 低域か高域を落とす（分離や録音の違い）
    if rng.random() < 0.5:
        if rng.random() < 0.5:
            b, a = butter(2, rng.uniform(3000.0, 12000.0) / (sr / 2))
        else:
            b, a = butter(2, rng.uniform(30.0, 150.0) / (sr / 2), btype="high")
        y = lfilter(b, a, y)
    # にじみ: ドラム以外のステムを混ぜる
    if bleed_files and rng.random() < 0.6:
        path = bleed_files[int(rng.integers(len(bleed_files)))]
        try:
            info = sf.info(str(path))
            dur = y.size / sr
            start = rng.uniform(0.0, max(info.duration - dur, 0.0))
            b_ = load_audio(path, sr, start, dur)
            if b_.size < y.size:
                b_ = np.pad(b_, (0, y.size - b_.size))
            b_ = b_[: y.size].astype(np.float64)
            ry = np.sqrt(np.mean(y**2)) + 1e-9
            rb = np.sqrt(np.mean(b_**2)) + 1e-9
            snr_db = rng.uniform(0.0, 18.0)
            y = y + b_ * (ry / rb) * 10 ** (-snr_db / 20)
        except (RuntimeError, sf.LibsndfileError):  # 壊れたファイルは飛ばす
            pass
    # 雑音
    if rng.random() < 0.3:
        y = y + rng.standard_normal(y.size) * (np.sqrt(np.mean(y**2)) + 1e-9) * 10 ** (rng.uniform(-40.0, -25.0) / 20)
    # 飽和
    if rng.random() < 0.3:
        k = rng.uniform(1.0, 4.0)
        y = np.tanh(k * y) / k
    # 短い残響
    if rng.random() < 0.3:
        rt = rng.uniform(0.1, 0.5)
        n = int(rt * sr)
        ir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / sr / rt)
        ir /= np.sqrt(np.sum(ir**2)) + 1e-9
        y = y + rng.uniform(0.1, 0.3) * fftconvolve(y, ir)[: y.size]
    peak = np.max(np.abs(y)) + 1e-9
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)


class EGMDChunks:
    """E-GMD のトラックから、決まった長さの区間を切り出して (対数メル, 打点のフレーム) を返す。

    `torch.utils.data.Dataset` として使う（`__len__`／`__getitem__`）。各エポックの区間は `set_epoch` で変わる。
    """

    def __init__(self, tracks: list[Track], cfg: FeatureConfig, chunk_sec: float = 8.0, chunks_per_track: int = 2,
                 bleed_files: list[Path] | None = None, do_augment: bool = True, classes: tuple[str, ...] = CLASSES5, seed: int = 0):
        self.tracks = tracks
        self.cfg = cfg
        self.chunk_sec = chunk_sec
        self.chunks_per_track = chunks_per_track
        self.bleed_files = bleed_files or []
        self.do_augment = do_augment
        self.classes = classes
        self.seed = seed
        self.epoch = 0
        self._hits_cache: dict[Path, list[Hit]] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.tracks) * self.chunks_per_track

    def hits_of(self, tr: Track) -> list[Hit]:
        if tr.midi not in self._hits_cache:
            self._hits_cache[tr.midi] = load_hits_midi(tr.midi)
        return self._hits_cache[tr.midi]

    def __getitem__(self, i: int):
        import torch

        rng = np.random.default_rng((self.seed * 1_000_003 + self.epoch * 7_919 + i) % (2**32))
        tr = self.tracks[i % len(self.tracks)]
        dur = tr.duration or sf.info(str(tr.audio)).duration
        start = float(rng.uniform(0.0, max(dur - self.chunk_sec, 0.0)))
        x = load_audio(tr.audio, self.cfg.sr, start, self.chunk_sec)
        n = int(self.chunk_sec * self.cfg.sr)
        x = np.pad(x, (0, max(n - x.size, 0)))[:n]
        hits = [Hit(h.time - start, h.cls, h.velocity) for h in self.hits_of(tr) if 0.0 <= h.time - start < self.chunk_sec]
        if self.do_augment:
            x = augment(x, self.cfg.sr, rng, self.bleed_files)
        feat = log_mel(x, self.cfg)
        y = frame_targets(hits, feat.shape[0], self.cfg, self.classes)
        return torch.from_numpy(feat), torch.from_numpy(y)
