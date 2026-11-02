"""小さな CRNN（採譜）。PyTorch。

入力 [batch, frames, n_mels] → 畳み込み2段（周波数方向にプール）→ 双方向 GRU → 各フレームの各種類の確率。
ONNX に書き出せる演算だけを使う。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DrumCRNN(nn.Module):
    """`pools`: 周波数方向に半分にする回数（1 か 2）。`delta`: 対数メルの時間差分（正の部分 = 立ち上がり）を 2 つ目の
    チャンネルとしてモデルの中で作る。小さい打の立ち上がりを拾いやすくする。ONNX の入力は対数メルのままで変わらない。"""

    def __init__(self, n_mels: int = 64, n_classes: int = 3, hidden: int = 48, pools: int = 2, delta: bool = False):
        super().__init__()
        self.delta = delta
        in_ch = 2 if delta else 1
        layers: list[nn.Module] = [nn.Conv2d(in_ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU()]
        if pools >= 2:
            layers.append(nn.MaxPool2d((1, 2)))  # 周波数方向だけ半分に
        layers += [nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((1, 2))]
        self.conv = nn.Sequential(*layers)
        self.rnn = nn.GRU(32 * (n_mels // (2 ** max(pools, 1))), hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, M]
        inp = x.unsqueeze(1)  # [B, 1, T, M]
        if self.delta:
            d = torch.relu(x[:, 1:, :] - x[:, :-1, :])  # 立ち上がりだけ
            d = torch.cat([torch.zeros_like(x[:, :1, :]), d], dim=1)
            inp = torch.cat([inp, d.unsqueeze(1)], dim=1)  # [B, 2, T, M]
        h = self.conv(inp)  # [B, C, T, M/4]
        b, c, t, m = h.shape
        h = h.permute(0, 2, 1, 3).reshape(b, t, c * m)
        h, _ = self.rnn(h)
        return torch.sigmoid(self.out(h))  # [B, T, classes]
