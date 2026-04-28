from __future__ import annotations

import torch
import torch.nn as nn


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvNormAct(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class TFEvidenceEncoder(nn.Module):
    """Dense time-frequency encoder for log-normalized VQT magnitude input."""

    def __init__(
        self,
        base_channels: int = 64,
        out_channels: int = 256,
        blocks_per_stage: int = 2,
    ) -> None:
        super().__init__()
        mid_channels = base_channels * 2
        self.stem = nn.Sequential(
            ConvNormAct(1, base_channels),
            *[ResidualBlock(base_channels) for _ in range(blocks_per_stage)],
        )
        self.down1 = nn.Sequential(
            ConvNormAct(base_channels, mid_channels, stride=2),
            *[ResidualBlock(mid_channels) for _ in range(blocks_per_stage)],
        )
        self.down2 = nn.Sequential(
            ConvNormAct(mid_channels, out_channels, stride=2),
            *[ResidualBlock(out_channels) for _ in range(blocks_per_stage)],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.down1(h)
        return self.down2(h)


SepFPEncoder = TFEvidenceEncoder
