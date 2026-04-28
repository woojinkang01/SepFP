from __future__ import annotations

import torch

from sepfp.data.batch_types import BranchContext, STEM_ORDER


def build_positive_masks(
    art_ctx: BranchContext,
    ref_ctx: BranchContext,
    stems: tuple[str, ...] = STEM_ORDER,
) -> dict[str, torch.BoolTensor]:
    masks: dict[str, torch.BoolTensor] = {}
    for stem_idx, stem in enumerate(stems):
        art_indices = torch.nonzero(art_ctx.active_mask[:, stem_idx], as_tuple=False).flatten()
        ref_indices = torch.nonzero(ref_ctx.active_mask[:, stem_idx], as_tuple=False).flatten()
        if art_indices.numel() == 0 or ref_indices.numel() == 0:
            continue

        pos_mask = torch.zeros((art_indices.numel(), ref_indices.numel()), dtype=torch.bool)
        for art_row, art_sample_idx in enumerate(art_indices.tolist()):
            art_tokens = set(art_ctx.provenance[art_sample_idx].get(stem, ()))
            for ref_col, ref_sample_idx in enumerate(ref_indices.tolist()):
                ref_tokens = set(ref_ctx.provenance[ref_sample_idx].get(stem, ()))
                pos_mask[art_row, ref_col] = len(art_tokens & ref_tokens) > 0

        masks[stem] = pos_mask
    return masks
