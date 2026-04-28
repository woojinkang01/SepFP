from __future__ import annotations

import torch
import torch.nn as nn

from sepfp.data.batch_types import SeparationLossOutput, StemBatch


class SeparationLoss(nn.Module):
    """L1 loss for predicted and target linear-magnitude VQT tensors."""

    def forward(
        self,
        pred_by_stem: dict[str, StemBatch],
        target_by_stem: dict[str, StemBatch],
    ) -> SeparationLossOutput:
        losses = []
        per_stem_loss = {}
        per_stem_count = {}
        for stem, pred_batch in pred_by_stem.items():
            if stem not in target_by_stem:
                continue
            target_batch = target_by_stem[stem]
            per_pair = torch.abs(pred_batch.tensor - target_batch.tensor).mean(dim=(1, 2, 3))
            losses.append(per_pair)
            per_stem_loss[stem] = per_pair.mean().detach()
            per_stem_count[stem] = int(per_pair.numel())

        if not losses:
            device = next(iter(pred_by_stem.values())).tensor.device if pred_by_stem else torch.device("cpu")
            return SeparationLossOutput(loss=torch.zeros((), device=device), per_stem_loss={}, per_stem_count={})
        return SeparationLossOutput(
            loss=torch.cat(losses, dim=0).mean(),
            per_stem_loss=per_stem_loss,
            per_stem_count=per_stem_count,
        )
