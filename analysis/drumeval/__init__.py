"""Phase 0 の採点ツール。

- `labels`: 打点ラベル（CSV／MIDI）の読み書きと種類の正規化
- `metrics`: 採譜の F 値（mir_eval）と多重解像度 STFT 距離
"""

from .labels import Hit, load_hits, save_hits_csv, COARSE_CLASSES, to_coarse
from .metrics import (
    transcription_scores,
    m0_transcription_score,
    multi_resolution_stft_distance,
)

__all__ = [
    "Hit",
    "load_hits",
    "save_hits_csv",
    "COARSE_CLASSES",
    "to_coarse",
    "transcription_scores",
    "m0_transcription_score",
    "multi_resolution_stft_distance",
]
