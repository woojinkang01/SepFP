from __future__ import annotations

import torch
import torch.nn as nn

from sepfp.data.batch_types import STEM_ORDER, StemBatch


class SourceQueryEvidenceExtractor(nn.Module):
    """Extract active stem evidence maps from shared time-frequency memory."""

    def __init__(
        self,
        stems: tuple[str, ...] = STEM_ORDER,
        in_channels: int = 256,
        evidence_channels: int = 192,
        query_dim: int = 256,
        num_attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.stems = stems
        self.stem_query = nn.Embedding(len(stems), query_dim)
        self.memory_proj = nn.Conv2d(in_channels, query_dim, kernel_size=1, bias=False)
        self.attn = nn.MultiheadAttention(query_dim, num_attention_heads, batch_first=True)
        self.evidence_proj = nn.Sequential(
            nn.Conv2d(in_channels, evidence_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(evidence_channels),
            nn.SiLU(inplace=True),
        )
        self.gate_from_memory = nn.Conv2d(query_dim, evidence_channels, kernel_size=1)
        self.film = nn.Linear(query_dim, evidence_channels * 2)
        self.out = nn.Sequential(
            nn.Conv2d(evidence_channels, evidence_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(evidence_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, memory: torch.Tensor, active_mask: torch.BoolTensor) -> dict[str, StemBatch]:
        active_mask = active_mask.to(memory.device)
        memory_tokens = self.memory_proj(memory)
        batch, channels, height, width = memory_tokens.shape
        tokens = memory_tokens.flatten(2).transpose(1, 2)
        out: dict[str, StemBatch] = {}

        for stem_idx, stem in enumerate(self.stems):
            sample_idx = torch.nonzero(active_mask[:, stem_idx], as_tuple=False).flatten()
            if sample_idx.numel() == 0:
                continue

            stem_tokens = tokens.index_select(0, sample_idx)
            query = self.stem_query.weight[stem_idx].view(1, 1, -1).expand(sample_idx.numel(), 1, -1)
            attended, _ = self.attn(query, stem_tokens, stem_tokens, need_weights=False)
            state = attended.squeeze(1)

            stem_memory = memory.index_select(0, sample_idx)
            stem_memory_tokens = memory_tokens.index_select(0, sample_idx)
            evidence = self.evidence_proj(stem_memory)
            gate = torch.sigmoid(self.gate_from_memory(stem_memory_tokens))
            gamma_beta = self.film(state).view(sample_idx.numel(), 2, -1, 1, 1)
            gamma, beta = gamma_beta[:, 0], gamma_beta[:, 1]
            u = self.out(evidence * gate * (1 + gamma) + beta)
            out[stem] = StemBatch(sample_idx=sample_idx, tensor=u)

        _ = batch, channels, height, width
        return out
