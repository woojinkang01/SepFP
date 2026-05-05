from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from sepfp.data.batch_types import BranchContext, STEM_ORDER, StemBatch
from sepfp.models.encoder import TFEvidenceEncoder
from sepfp.models.projector import EvidenceProjector
from sepfp.models.sep_decoder import LinearMagMaskDecoder
from sepfp.models.stem_head import SourceQueryEvidenceExtractor


@dataclass(frozen=True)
class BranchOutputs:
    features: torch.Tensor
    stem_latents: dict[str, StemBatch]
    stem_preds: dict[str, StemBatch]
    stem_embeds: dict[str, StemBatch]


class SepFPModel(nn.Module):
    VALID_MASK_MODES = {"active_softmax", "independent_sigmoid", "independent_softplus", "independent_capped"}
    VALID_ASID_GRADIENT_ROUTES = {"projector_only", "evidence", "full"}

    def __init__(
        self,
        stems: tuple[str, ...] = STEM_ORDER,
        encoder_channels: int = 256,
        evidence_channels: int = 192,
        query_dim: int = 256,
        attention_heads: int = 4,
        decoder_hidden_channels: int = 128,
        projector_hidden_channels: int = 256,
        projector_out_dim: int = 512,
        mask_mode: str = "active_softmax",
        max_mask: float = 2.0,
        asid_gradient_route: str = "projector_only",
    ) -> None:
        super().__init__()
        if mask_mode not in self.VALID_MASK_MODES:
            raise ValueError(f"Unknown mask_mode: {mask_mode}")
        if asid_gradient_route not in self.VALID_ASID_GRADIENT_ROUTES:
            raise ValueError(f"Unknown asid_gradient_route: {asid_gradient_route}")
        self.stems = stems
        self.mask_mode = mask_mode
        self.max_mask = max_mask
        self.asid_gradient_route = asid_gradient_route
        self.encoder = TFEvidenceEncoder(out_channels=encoder_channels)
        self.evidence = SourceQueryEvidenceExtractor(
            stems=stems,
            in_channels=encoder_channels,
            evidence_channels=evidence_channels,
            query_dim=query_dim,
            num_attention_heads=attention_heads,
        )
        self.decoder = LinearMagMaskDecoder(
            in_channels=evidence_channels,
            hidden_channels=decoder_hidden_channels,
        )
        self.projectors = nn.ModuleDict(
            {
                stem: EvidenceProjector(
                    in_channels=evidence_channels,
                    hidden_channels=projector_hidden_channels,
                    out_dim=projector_out_dim,
                )
                for stem in stems
            }
        )

    def _active_softmax_masks(
        self,
        logits_by_stem: dict[str, StemBatch],
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        masks_by_stem = {stem: torch.zeros_like(batch.tensor) for stem, batch in logits_by_stem.items()}

        for sample_idx in range(batch_size):
            entries: list[tuple[str, int, torch.Tensor]] = []
            for stem, batch in logits_by_stem.items():
                row = torch.nonzero(batch.sample_idx == sample_idx, as_tuple=False).flatten()
                if row.numel() == 0:
                    continue
                row_idx = int(row[0].item())
                entries.append((stem, row_idx, batch.tensor[row_idx]))

            if not entries:
                continue

            stacked = torch.stack([entry[2] for entry in entries], dim=0)
            normalized = torch.softmax(stacked, dim=0)
            for normalized_idx, (stem, row_idx, _) in enumerate(entries):
                masks_by_stem[stem][row_idx] = normalized[normalized_idx]

        return masks_by_stem

    def _independent_masks(self, logits_by_stem: dict[str, StemBatch]) -> dict[str, torch.Tensor]:
        if self.mask_mode == "independent_sigmoid":
            return {stem: torch.sigmoid(batch.tensor) for stem, batch in logits_by_stem.items()}
        if self.mask_mode == "independent_softplus":
            return {stem: F.softplus(batch.tensor) for stem, batch in logits_by_stem.items()}
        if self.mask_mode == "independent_capped":
            return {stem: self.max_mask * torch.sigmoid(batch.tensor) for stem, batch in logits_by_stem.items()}
        raise ValueError(f"Unknown mask_mode: {self.mask_mode}")

    def _evidence_for_asid(self, features: torch.Tensor, active_mask: torch.BoolTensor) -> dict[str, StemBatch]:
        if not self.training:
            return self.evidence(features.detach(), active_mask)

        params = dict(self.evidence.named_parameters())
        buffers = {name: buffer.detach().clone() for name, buffer in self.evidence.named_buffers()}
        return functional_call(self.evidence, (params, buffers), (features.detach(), active_mask))

    def forward_branch(self, ctx: BranchContext, *, compute_separation: bool = True) -> BranchOutputs:
        features = self.encoder(ctx.x_input)
        if not compute_separation and self.asid_gradient_route == "evidence":
            stem_latents = self.evidence(features.detach(), ctx.active_mask)
            asid_stem_latents = stem_latents
            asid_projector_detach = False
        else:
            stem_latents = self.evidence(features, ctx.active_mask)
            asid_stem_latents = stem_latents
            asid_projector_detach = True
            if self.asid_gradient_route == "evidence":
                asid_stem_latents = self._evidence_for_asid(features, ctx.active_mask)
                asid_projector_detach = False
            elif self.asid_gradient_route == "full":
                asid_projector_detach = False

        logits_by_stem: dict[str, StemBatch] = {}
        stem_embeds: dict[str, StemBatch] = {}
        for stem, stem_batch in stem_latents.items():
            asid_stem_batch = asid_stem_latents[stem]
            z = self.projectors[stem](asid_stem_batch.tensor, detach_input=asid_projector_detach)
            if compute_separation:
                logits = self.decoder(stem_batch.tensor)
                logits_by_stem[stem] = StemBatch(sample_idx=stem_batch.sample_idx, tensor=logits)
            stem_embeds[stem] = StemBatch(
                sample_idx=asid_stem_batch.sample_idx,
                tensor=z,
                provenance=tuple(ctx.provenance[idx].get(stem, ()) for idx in asid_stem_batch.sample_idx.tolist()),
            )

        stem_preds: dict[str, StemBatch] = {}
        if compute_separation:
            if self.mask_mode == "active_softmax":
                masks_by_stem = self._active_softmax_masks(logits_by_stem, batch_size=ctx.x_input.size(0))
            else:
                masks_by_stem = self._independent_masks(logits_by_stem)

            for stem, stem_batch in logits_by_stem.items():
                sample_idx = stem_batch.sample_idx
                carrier = ctx.x_linear_mag.to(stem_batch.tensor.device).index_select(0, sample_idx)
                mask = masks_by_stem[stem]
                pred = mask * carrier
                stem_preds[stem] = StemBatch(
                    sample_idx=sample_idx,
                    tensor=pred,
                    provenance=tuple(ctx.provenance[idx].get(stem, ()) for idx in sample_idx.tolist()),
                    extras={
                        "mask": mask.detach(),
                        "mask_logits": stem_batch.tensor.detach(),
                        "domain": "linear_mag",
                    },
                )

        return BranchOutputs(
            features=features,
            stem_latents=stem_latents,
            stem_preds=stem_preds,
            stem_embeds=stem_embeds,
        )
