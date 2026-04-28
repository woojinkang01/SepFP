from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearMagMaskDecoder(nn.Module):
    """Predict linear-magnitude VQT mask logits from evidence only."""

    def __init__(self, in_channels: int = 192, hidden_channels: int = 128) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.SiLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(hidden_channels // 2, hidden_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 4),
            nn.SiLU(inplace=True),
        )
        self.out = nn.Conv2d(hidden_channels // 4, 1, kernel_size=1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        h = self.block1(u)
        h = F.interpolate(h, size=(126, 128), mode="bilinear", align_corners=False)
        h = self.block2(h)
        h = F.interpolate(h, size=(252, 256), mode="bilinear", align_corners=False)
        h = self.block3(h)
        return self.out(h)
