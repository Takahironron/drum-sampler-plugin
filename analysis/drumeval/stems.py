"""評価用の実ステム（`data/stems/<曲ID>/`）の探索と、疑似ラベルでの学習用の区間の切り出し。

曲フォルダの中身（`run_eval.py` と同じ約束）:
    drums.wav | drum.wav | <曲名>_ドラム.wav   分離済みのドラムのステム
    キック.wav スネア.wav ハイハット.wav ...     分離ツールのパート別ステム（疑似正解の出どころ）
    hits.csv | hits.mid                          本当の正解（区分 B）
    hits_pseudo.csv                              `run_eval.py` が書く疑似正解。先頭のコメント行に評価の開始位置（start_sec）、
                                                 区間の長さ（seconds）、ラベルのあるパート（parts）
    start.txt                                    評価の開始位置の上書き（秒）

`StemChunks` は、採譜モデルを実ステムに合わせる微調整（`train_transcriber.py --finetune-stems`）のデータセット。
疑似ラベルは分離ツールのパート別ステムから作るので、パートの無い種類（例: タムのステムが無い曲のタム）は正解が無い。
その種類は損失から外す（`__getitem__` が返すマスク）。CD 曲のステムで学習したモデルは評価用に限る（R3）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from .egmd import CLASSES5, augment, load_audio
from .features import FeatureConfig, frame_targets, log_mel
from .labels import Hit, load_hits, read_hits_csv_meta, to_coarse

AUDIO_EXT = (".wav", ".flac", ".aif", ".aiff", ".ogg")

# パート別に分離されたステムの名前（SpectraLayers の日本語名と英語名）→ 種類
PART_NAMES = {
    "キック": "kick", "kick": "kick", "kicks": "kick",
    "スネア": "snare", "snare": "snare",
    "ハイハット": "hihat", "hihat": "hihat", "hi-hat": "hihat", "hats": "hihat",
    "シンバル": "cymbal", "cymbal": "cymbal", "cymbals": "cymbal",
    "タム": "tom", "tom": "tom", "toms": "tom",
}


def find_stem(song_dir: Path) -> Path | None:
    """`drums.wav`／`drums.flac` を優先し、無ければフォルダ内の音声ファイルが 1 つだけならそれを使う。"""
    for name in ("drums.wav", "drums.flac", "drum.wav", "drum.flac", "ドラム.wav"):
        if (song_dir / name).exists():
            return song_dir / name
    # SpectraLayers の書き出し「<曲名>_ドラム.wav」
    sl = [p for p in song_dir.iterdir() if p.suffix.lower() in AUDIO_EXT and p.stem.endswith("_ドラム")]
    if len(sl) == 1:
        return sl[0]
    audio = [p for p in song_dir.iterdir() if p.suffix.lower() in AUDIO_EXT and not p.name.startswith("rerender")]
    return audio[0] if len(audio) == 1 else None


def find_part_stems(song_dir: Path) -> dict[str, Path]:
    """パート別ステム（キック.wav など）があれば 種類 → ファイル を返す。"""
    out: dict[str, Path] = {}
    for p in song_dir.iterdir():
        if p.suffix.lower() not in AUDIO_EXT:
            continue
        key = p.stem.lower()
        for name, cls in PART_NAMES.items():
            if key == name.lower() or key.endswith("_" + name.lower()):
                out[cls] = p
    return out


def detect_drum_start(x: np.ndarray, sr: int, win_sec: float = 1.0, rel_db: float = -15.0, hold: int = 3) -> float:
    """ドラムが鳴り始める位置（秒）。1 秒窓の RMS が最大から rel_db 以内に `hold` 窓続けて入る最初の窓。見つからなければ 0。

    混ざったドラムのステムには他の楽器の残りが入るので、できればキックとスネアのパート別ステムの合計を渡す（`resolve_start` が行う）。
    確実なのは曲フォルダの `start.txt`。
    """
    n = int(win_sec * sr)
    if x.size < n * (hold + 1):
        return 0.0
    k = x.size // n
    rms = np.sqrt(np.mean(x[: k * n].reshape(k, n) ** 2, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms)
    thr = db.max() + rel_db
    above = db > thr
    for i in range(k - hold + 1):
        if above[i : i + hold].all():
            return i * win_sec
    return 0.0


def read_start_override(song_dir: Path) -> float | None:
    """曲フォルダの `start.txt`（秒を 1 つ書く）があれば、その位置から評価する。"""
    p = song_dir / "start.txt"
    if p.exists():
        try:
            return float(p.read_text(encoding="utf-8").strip())
        except ValueError:
            return None
    return None


def _load_mono_at(path: Path, sr: int) -> np.ndarray:
    return load_audio(path, sr)


def resolve_start(song_dir: Path, x_full: np.ndarray, sr: int, start_sec: float | None = None,
                  load: Callable[[Path, int], np.ndarray] = _load_mono_at) -> float:
    """評価の開始位置: `start.txt` → `start_sec` → ドラムが鳴り始める位置の自動検出（キックとスネアのパート別ステムがあればその合計で）。

    `load(path, sr)` はパート別ステムをモノラル・`sr` で読む関数（呼び出し側の読み方に合わせられる）。
    """
    start = read_start_override(song_dir)
    if start is not None:
        return start
    if start_sec is not None:
        return start_sec
    parts = find_part_stems(song_dir)
    probe = x_full
    if "kick" in parts and "snare" in parts:
        k = load(parts["kick"], sr)
        s_ = load(parts["snare"], sr)
        n = min(k.size, s_.size)
        probe = k[:n] + s_[:n]
    return detect_drum_start(probe, sr)


# ---------------------------------------------------------------------------
# 疑似ラベル付きの実ステム（微調整の学習データ）
# ---------------------------------------------------------------------------

@dataclass
class StemSong:
    song_id: str
    audio: Path
    hits: list[Hit]  # `start` からの相対時刻
    start: float  # 評価（ラベル）の区間の開始位置（秒、ファイルの先頭から）
    seconds: float  # ラベルのある区間の長さ（秒）
    labeled: tuple[str, ...]  # 正解のある種類（この種類だけ損失に入れる）
    truth: str  # "hits" (本当の正解) か "pseudo"


def read_stem_songs(stems_dir: str | Path, ids: list[str] | None = None, max_seconds: float = 60.0) -> list[StemSong]:
    """`stems_dir` の曲フォルダのうち、正解（`hits.csv`／`hits.mid`）か疑似正解（`hits_pseudo.csv`）のあるものを返す。

    疑似正解の区間（開始位置と長さ）は CSV の先頭のコメント行から取る。古い CSV（コメント行なし）は `resolve_start` と
    `max_seconds` で `run_eval.py` の既定と同じ区間にする。ラベルのある種類は、本当の正解なら 5 種すべて、
    疑似正解ならパート別ステムのある種類（無ければ CSV に出てくる種類）。
    """
    stems_dir = Path(stems_dir)
    songs: list[StemSong] = []
    for d in sorted(p for p in stems_dir.iterdir() if p.is_dir()):
        if ids and d.name not in ids:
            continue
        audio = find_stem(d)
        if audio is None:
            continue
        truth = next((p for p in (d / "hits.csv", d / "hits.mid", d / "hits.midi") if p.exists()), None)
        pseudo = d / "hits_pseudo.csv"
        if truth is not None:
            hits = load_hits(truth)
            start = read_start_override(d) or 0.0
            dur = sf.info(str(audio)).duration - start
            hits = [Hit(h.time - start, h.cls, h.velocity) for h in hits if h.time >= start]
            if max_seconds and dur > max_seconds:
                dur = max_seconds
            hits = [h for h in hits if h.time < dur]
            songs.append(StemSong(d.name, audio, hits, start, dur, CLASSES5, "hits"))
        elif pseudo.exists():
            meta = read_hits_csv_meta(pseudo)
            hits = load_hits(pseudo)
            if "start_sec" in meta:
                start = float(meta["start_sec"])
            else:
                x_full = load_audio(audio, 22050)
                start = resolve_start(d, x_full, 22050)
            total = sf.info(str(audio)).duration - start
            dur = float(meta["seconds"]) if "seconds" in meta else (min(total, max_seconds) if max_seconds else total)
            if "parts" in meta:
                labeled = tuple(c for c in CLASSES5 if c in meta["parts"].split(","))
            else:
                labeled = tuple(sorted(find_part_stems(d))) or tuple(sorted({to_coarse(h.cls) for h in hits}))
            labeled = tuple(c for c in CLASSES5 if c in labeled)
            songs.append(StemSong(d.name, audio, [h for h in hits if h.time < dur], start, dur, labeled, "pseudo"))
    return songs


class StemChunks:
    """実ステムの曲から決まった長さの区間を切り出し、(対数メル, 打点のフレーム, 種類のマスク) を返す。

    `torch.utils.data.Dataset` として使う。マスクは [classes] の 0/1 で、ラベルの無い種類（パートの無い種類）が 0。
    区間はラベルのある範囲（`StemSong.start` から `seconds`）の中から乱数で選ぶ。音声は曲ごとに一度だけ読んで持つ。
    """

    def __init__(self, songs: list[StemSong], cfg: FeatureConfig, chunk_sec: float = 8.0, chunks_per_song: int = 40,
                 bleed_files: list[Path] | None = None, do_augment: bool = True, classes: tuple[str, ...] = CLASSES5, seed: int = 0):
        self.songs = songs
        self.cfg = cfg
        self.chunk_sec = chunk_sec
        self.chunks_per_song = chunks_per_song
        self.bleed_files = bleed_files or []
        self.do_augment = do_augment
        self.classes = classes
        self.seed = seed
        self.epoch = 0
        self._audio: dict[str, np.ndarray] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.songs) * self.chunks_per_song

    def audio_of(self, song: StemSong) -> np.ndarray:
        if song.song_id not in self._audio:
            self._audio[song.song_id] = load_audio(song.audio, self.cfg.sr, song.start, song.seconds)
        return self._audio[song.song_id]

    def mask_of(self, song: StemSong) -> np.ndarray:
        return np.array([1.0 if c in song.labeled else 0.0 for c in self.classes], dtype=np.float32)

    def __getitem__(self, i: int):
        import torch

        rng = np.random.default_rng((self.seed * 1_000_003 + self.epoch * 7_919 + 104_729 + i) % (2**32))
        song = self.songs[i % len(self.songs)]
        full = self.audio_of(song)
        n = int(self.chunk_sec * self.cfg.sr)
        offset = float(rng.uniform(0.0, max(full.size / self.cfg.sr - self.chunk_sec, 0.0)))
        a = int(offset * self.cfg.sr)
        x = np.pad(full[a : a + n], (0, max(n - full[a : a + n].size, 0)))
        hits = [Hit(h.time - offset, h.cls, h.velocity) for h in song.hits if 0.0 <= h.time - offset < self.chunk_sec]
        if self.do_augment:
            x = augment(x.astype(np.float32), self.cfg.sr, rng, self.bleed_files)
        feat = log_mel(x, self.cfg)
        y = frame_targets(hits, feat.shape[0], self.cfg, self.classes)
        return torch.from_numpy(feat), torch.from_numpy(y), torch.from_numpy(self.mask_of(song))


def song_examples(songs: list[StemSong], cfg: FeatureConfig) -> list[tuple[np.ndarray, list[Hit]]]:
    """各曲のラベルのある区間を丸ごと (対数メル, 打点) にする（微調整の前後の採点用）。"""
    out = []
    for s in songs:
        x = load_audio(s.audio, cfg.sr, s.start, s.seconds)
        out.append((log_mel(x, cfg), list(s.hits)))
    return out
