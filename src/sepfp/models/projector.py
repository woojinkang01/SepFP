from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def detach_evidence_for_asid(u: torch.Tensor) -> torch.Tensor:
    return u.detach()


class AttentionPool2d(nn.Module):
    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        weights = self.score(x).flatten(2)
        weights = torch.softmax(weights, dim=-1)
        values = x.flatten(2)
        return torch.bmm(values, weights.transpose(1, 2)).view(batch, channels)


class EvidenceProjector(nn.Module):
    def __init__(
        self,
        in_channels: int = 192,
        hidden_channels: int = 256,
        out_dim: int = 512,
    ) -> None:
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.pool = AttentionPool2d(hidden_channels, hidden_channels)
        self.proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, out_dim, bias=False),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        h = self.pre(detach_evidence_for_asid(u))
        z = self.proj(self.pool(h))
        return F.normalize(z, dim=-1)
