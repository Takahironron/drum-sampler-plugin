"""小さな CRNN（採譜）。PyTorch。

入力 [batch, frames, n_mels] → 畳み込み2段（周波数方向にプール）→ 双方向 GRU → 各フレームの各種類の確率。
ONNX に書き出せる演算だけを使う。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DrumCRNN(nn.Module):
    def __init__(self, n_mels: int = 64, n_classes: int = 3, hidden: int = 48):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # 周波数方向だけ半分に
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((1, 2)),
        )
        self.rnn = nn.GRU(32 * (n_mels // 4), hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, M]
        h = self.conv(x.unsqueeze(1))  # [B, C, T, M/4]
        b, c, t, m = h.shape
        h = h.permute(0, 2, 1, 3).reshape(b, t, c * m)
        h, _ = self.rnn(h)
        return torch.sigmoid(self.out(h))  # [B, T, classes]
