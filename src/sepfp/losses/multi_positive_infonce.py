from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sepfp.data.batch_types import InfoNCELossOutput, StemBatch


class MultiPositiveInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.01, trainable: bool = True) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(temperature).log(), requires_grad=trainable)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(
        self,
        art_z_by_stem: dict[str, StemBatch],
        ref_z_by_stem: dict[str, StemBatch],
        pos_mask_by_stem: dict[str, torch.BoolTensor],
    ) -> InfoNCELossOutput:
        stem_losses_for_grad: dict[str, torch.Tensor] = {}
        per_stem_loss: dict[str, torch.Tensor] = {}
        per_stem_anchor_count: dict[str, int] = {}
        skipped_anchor_count = 0
        device = None

        for stem, art_batch in art_z_by_stem.items():
            if stem not in ref_z_by_stem or stem not in pos_mask_by_stem:
                continue
            ref_batch = ref_z_by_stem[stem]
            pos_mask = pos_mask_by_stem[stem].to(art_batch.tensor.device)
            device = art_batch.tensor.device

            art_z = F.normalize(art_batch.tensor, dim=-1)
            ref_z = F.normalize(ref_batch.tensor, dim=-1)
            logits = art_z @ ref_z.transpose(0, 1) / self.temperature

            stem_losses = []
            valid_anchor_count = 0
            for anchor_idx in range(logits.size(0)):
                positives = pos_mask[anchor_idx]
                n_pos = int(positives.sum().item())
                n_ref = logits.size(1)
                if n_pos == 0 or n_ref <= n_pos:
                    skipped_anchor_count += 1
                    continue

                numerator = torch.logsumexp(logits[anchor_idx][positives], dim=0)
                denominator = torch.logsumexp(logits[anchor_idx], dim=0)
                stem_losses.append(-(numerator - denominator))
                valid_anchor_count += 1

            if valid_anchor_count == 0:
                continue

            stem_loss = torch.stack(stem_losses).mean()
            stem_losses_for_grad[stem] = stem_loss
            per_stem_loss[stem] = stem_loss.detach()
            per_stem_anchor_count[stem] = valid_anchor_count

        n_anchor = sum(per_stem_anchor_count.values())
        n_stem = len(stem_losses_for_grad)
        if n_stem == 0:
            zero = torch.zeros((), device=device or torch.device("cpu"))
            return InfoNCELossOutput(
                loss=zero,
                n_anchor=0,
                per_stem_loss={},
                per_stem_anchor_count={},
                skipped_anchor_count=skipped_anchor_count,
            )

        return InfoNCELossOutput(
            loss=torch.stack(tuple(stem_losses_for_grad.values())).mean(),
            n_anchor=n_anchor,
            per_stem_loss=per_stem_loss,
            per_stem_anchor_count=per_stem_anchor_count,
            skipped_anchor_count=skipped_anchor_count,
        )
