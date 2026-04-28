from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


STEM_ORDER = ("vocals", "drums", "bass", "guitar", "piano", "others")


@dataclass
class EffectOp:
    name: str
    params: dict[str, float | int]


@dataclass
class BranchEffectParams:
    ops: tuple[EffectOp, ...] = ()


@dataclass
class StemSource:
    audio: torch.Tensor
    provenance_id: str


@dataclass
class SepFPRawExample:
    mix_A: torch.Tensor
    mix_B: torch.Tensor
    mix_AB: torch.Tensor
    stem_types_A: tuple[str, ...]
    stem_types_B: tuple[str, ...]
    stem_types_AB: tuple[str, ...]
    individual_stems_A: dict[str, tuple[StemSource, ...]]
    individual_stems_B: dict[str, tuple[StemSource, ...]]
    individual_stems_AB: dict[str, tuple[StemSource, ...]]
    effect_params_A: BranchEffectParams
    effect_params_B: BranchEffectParams
    effect_params_AB: BranchEffectParams
    song_id: str
    frame_offset: int
    partition_indices_A: tuple[int, ...]
    partition_indices_B: tuple[int, ...]
    partition_indices_AB: tuple[int, ...]
    provenance_A: dict[str, tuple[str, ...]]
    provenance_B: dict[str, tuple[str, ...]]
    provenance_AB: dict[str, tuple[str, ...]]


@dataclass
class SepFPTrainBatch:
    mix_A: torch.Tensor
    mix_B: torch.Tensor
    mix_AB: torch.Tensor
    stem_types_A: tuple[tuple[str, ...], ...]
    stem_types_B: tuple[tuple[str, ...], ...]
    stem_types_AB: tuple[tuple[str, ...], ...]
    individual_stems_A: tuple[dict[str, tuple[StemSource, ...]], ...]
    individual_stems_B: tuple[dict[str, tuple[StemSource, ...]], ...]
    individual_stems_AB: tuple[dict[str, tuple[StemSource, ...]], ...]
    effect_params_A: tuple[BranchEffectParams, ...]
    effect_params_B: tuple[BranchEffectParams, ...]
    effect_params_AB: tuple[BranchEffectParams, ...]
    song_ids: tuple[str, ...]
    frame_offsets: torch.Tensor
    partition_indices_A: tuple[tuple[int, ...], ...]
    partition_indices_B: tuple[tuple[int, ...], ...]
    partition_indices_AB: tuple[tuple[int, ...], ...]
    provenance_A: tuple[dict[str, tuple[str, ...]], ...]
    provenance_B: tuple[dict[str, tuple[str, ...]], ...]
    provenance_AB: tuple[dict[str, tuple[str, ...]], ...]


@dataclass
class BranchContext:
    name: str
    x_complex: torch.Tensor
    x_input: torch.Tensor
    x_linear_mag: torch.Tensor
    gain: torch.Tensor
    active_mask: torch.BoolTensor
    crop_meta: dict[str, torch.Tensor]
    provenance: tuple[dict[str, tuple[str, ...]], ...]
    effect_params: tuple[BranchEffectParams, ...]


@dataclass
class StemBatch:
    sample_idx: torch.LongTensor
    tensor: torch.Tensor
    provenance: tuple[tuple[str, ...], ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class StemLossBreakdown:
    loss: torch.Tensor
    count: int


@dataclass
class SeparationLossOutput:
    loss: torch.Tensor
    per_stem_loss: dict[str, torch.Tensor]
    per_stem_count: dict[str, int]


@dataclass
class InfoNCELossOutput:
    loss: torch.Tensor
    n_anchor: int
    per_stem_loss: dict[str, torch.Tensor]
    per_stem_anchor_count: dict[str, int]
    skipped_anchor_count: int
