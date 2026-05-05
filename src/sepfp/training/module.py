from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Type

import torch
import torch.distributed as dist
import torch.nn as nn
import lightning as L

from sepfp.compat import LightningModule
from sepfp.data import STEM_ORDER
from sepfp.data.batch_types import BranchContext, SepFPTrainBatch, StemBatch
from sepfp.data.preprocess import build_art_branch, build_ref_branch
from sepfp.data.provenance import build_positive_masks
from sepfp.data.targets import build_sep_targets
from sepfp.data.vqt import VQT
from sepfp.losses.multi_positive_infonce import MultiPositiveInfoNCELoss
from sepfp.losses.separation import SeparationLoss
from sepfp.models.sepfp_model import SepFPModel
from sepfp.training.optim import build_sepfp_optimizer


@dataclass(frozen=True)
class StepOutput:
    loss: torch.Tensor
    sep_loss: torch.Tensor
    asid_loss: torch.Tensor
    n_anchor: int
    per_stem_sep_loss: dict[str, torch.Tensor]
    per_stem_asid_loss: dict[str, torch.Tensor]
    per_stem_asid_anchor_count: dict[str, int]


class SepFPLightningModule(LightningModule):
    def __init__(
        self,
        model: SepFPModel | None = None,
        transform: nn.Module | None = None,
        sep_loss: nn.Module | None = None,
        asid_loss: MultiPositiveInfoNCELoss | None = None,
        optimizer: Callable | None = None,
        scheduler: Callable | None = None,
        sample_rate: int = 16000,
        block_size: tuple[int, int] = (252, 256),
        time_stretch: tuple[float, float] | None = (0.7, 1.5),
        pitch_shift: bool = True,
        lognorm_mean: float = 0.0,
        lognorm_std: float = 1.0,
        pitch_crop_bins: int = 18,
        stems: tuple[str, ...] = STEM_ORDER,
        lambda_sep: float = 100.0,
        lambda_asid_final: float = 1.0,
        lambda_asid_warmup_epochs: int = 5,
        apply_effects: Callable[[torch.Tensor, int, object], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.model = model or SepFPModel(stems=stems)
        self.transform = transform or VQT(output_format="Complex")
        self.sep_loss = sep_loss or SeparationLoss()
        self.asid_loss = asid_loss or MultiPositiveInfoNCELoss()
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.time_stretch = time_stretch
        self.pitch_shift = pitch_shift
        self.lognorm_mean = lognorm_mean
        self.lognorm_std = lognorm_std
        self.pitch_crop_bins = pitch_crop_bins
        self.stems = stems
        self.lambda_sep = lambda_sep
        self.lambda_asid_final = lambda_asid_final
        self.lambda_asid_warmup_epochs = lambda_asid_warmup_epochs
        self.apply_effects = apply_effects

    def _crop_size(self) -> int:
        return self.pitch_crop_bins

    def _merge_stem_batches(
        self,
        first: dict[str, StemBatch],
        second: dict[str, StemBatch],
    ) -> dict[str, StemBatch]:
        merged = dict(first)
        for stem, batch in second.items():
            if stem not in merged:
                merged[stem] = batch
                continue
            existing = merged[stem]
            merged[stem] = StemBatch(
                sample_idx=torch.cat([existing.sample_idx, batch.sample_idx], dim=0),
                tensor=torch.cat([existing.tensor, batch.tensor], dim=0),
                provenance=existing.provenance + batch.provenance,
            )
        return merged

    @staticmethod
    def _move_stem_batches(
        batches: dict[str, StemBatch],
        device: torch.device | str,
    ) -> dict[str, StemBatch]:
        return {
            stem: StemBatch(
                sample_idx=batch.sample_idx.to(device=device),
                tensor=batch.tensor.to(device=device),
                provenance=batch.provenance,
                extras=batch.extras,
            )
            for stem, batch in batches.items()
        }

    def _lambda_asid(self) -> float:
        if self.lambda_asid_warmup_epochs <= 0:
            return self.lambda_asid_final
        current_epoch = int(getattr(self, "current_epoch", 0))
        progress = min(float(current_epoch + 1) / self.lambda_asid_warmup_epochs, 1.0)
        return self.lambda_asid_final * progress

    def _log_optimizer_lrs(self, stage: str, on_step: bool, on_epoch: bool, batch_size: int) -> None:
        if stage != "train":
            return
        try:
            trainer = self.trainer
        except RuntimeError:
            return
        if trainer is None or not getattr(trainer, "optimizers", None):
            return
        for optimizer in trainer.optimizers:
            for group_idx, group in enumerate(optimizer.param_groups):
                group_name = group.get("name", f"group_{group_idx}")
                self.log(
                    f"{stage}/lr/{group_name}",
                    float(group["lr"]),
                    on_step=on_step,
                    on_epoch=on_epoch,
                    batch_size=batch_size,
                )

    @staticmethod
    def _distributed_sum(value: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            value = value.clone()
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    def _log_train_stem_metrics(
        self,
        stage: str,
        sep_output,
        asid_output,
        log_kwargs: dict[str, object],
    ) -> None:
        for stem, stem_sep_loss in sep_output.per_stem_loss.items():
            self.log(f"{stage}/loss_raw/sep/{stem}", stem_sep_loss, **log_kwargs)
        for stem, stem_count in sep_output.per_stem_count.items():
            self.log(f"{stage}/sep_count/{stem}", float(stem_count), **log_kwargs)
        for stem, stem_asid_loss in asid_output.per_stem_loss.items():
            self.log(f"{stage}/loss_raw/asid/{stem}", stem_asid_loss, **log_kwargs)
        for stem, anchor_count in asid_output.per_stem_anchor_count.items():
            self.log(f"{stage}/asid_anchor_count/{stem}", float(anchor_count), **log_kwargs)

    def _log_val_stem_metrics(
        self,
        stage: str,
        sep_output,
        asid_output,
        device: torch.device,
    ) -> None:
        # Validation metrics must be logged in a fixed order on every DDP rank.
        # Per-stem dictionaries are sparse because each rank can see different active stems.
        for stem in self.stems:
            sep_count = torch.tensor(float(sep_output.per_stem_count.get(stem, 0)), device=device)
            sep_loss = sep_output.per_stem_loss.get(stem)
            sep_sum = torch.zeros((), device=device) if sep_loss is None else sep_loss.detach() * sep_count
            sep_count_global = self._distributed_sum(sep_count)
            sep_sum_global = self._distributed_sum(sep_sum)
            sep_loss_global = sep_sum_global / sep_count_global.clamp_min(1.0)
            sep_batch_size = max(int(sep_count_global.detach().item()), 1)
            self.log(
                f"{stage}/loss_raw/sep/{stem}",
                sep_loss_global,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
                batch_size=sep_batch_size,
            )
            self.log(
                f"{stage}/sep_count/{stem}",
                sep_count_global,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
                reduce_fx="sum",
                batch_size=1,
            )

            asid_count = torch.tensor(float(asid_output.per_stem_anchor_count.get(stem, 0)), device=device)
            asid_loss = asid_output.per_stem_loss.get(stem)
            asid_sum = torch.zeros((), device=device) if asid_loss is None else asid_loss.detach() * asid_count
            asid_count_global = self._distributed_sum(asid_count)
            asid_sum_global = self._distributed_sum(asid_sum)
            asid_loss_global = asid_sum_global / asid_count_global.clamp_min(1.0)
            asid_batch_size = max(int(asid_count_global.detach().item()), 1)
            self.log(
                f"{stage}/loss_raw/asid/{stem}",
                asid_loss_global,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
                batch_size=asid_batch_size,
            )
            self.log(
                f"{stage}/asid_anchor_count/{stem}",
                asid_count_global,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
                reduce_fx="sum",
                batch_size=1,
            )

    def shared_step(self, batch: SepFPTrainBatch, stage: str = "train") -> StepOutput:
        with torch.no_grad():
            x_A_complex = self.transform(batch.mix_A)
            x_B_complex = self.transform(batch.mix_B)
            x_AB_complex = self.transform(batch.mix_AB)

            art_ctx: BranchContext = build_art_branch(
                batch=batch,
                x_A_complex=x_A_complex,
                x_B_complex=x_B_complex,
                block_size=self.block_size,
                mean=self.lognorm_mean,
                std=self.lognorm_std,
                pitch_shift=self.pitch_shift,
                crop_size=self._crop_size(),
                stems=self.stems,
            )
            ref_ctx: BranchContext = build_ref_branch(
                batch=batch,
                x_AB_complex=x_AB_complex,
                block_size=self.block_size,
                mean=self.lognorm_mean,
                std=self.lognorm_std,
                crop_size=self._crop_size(),
                time_stretch=self.time_stretch,
                stems=self.stems,
            )
            art_targets, ref_targets = build_sep_targets(
                batch=batch,
                art_ctx=art_ctx,
                ref_ctx=ref_ctx,
                vqt_transform=self.transform,
                apply_effects=self.apply_effects,
                sample_rate=self.sample_rate,
                block_size=self.block_size,
                mean=self.lognorm_mean,
                std=self.lognorm_std,
                stems=self.stems,
            )
            pos_masks = build_positive_masks(art_ctx, ref_ctx, stems=self.stems)
            art_targets = self._move_stem_batches(art_targets, device="cpu")
            ref_targets = self._move_stem_batches(ref_targets, device="cpu")

        art_out = self.model.forward_branch(art_ctx)
        ref_out = self.model.forward_branch(ref_ctx)
        target_device = art_ctx.x_input.device
        art_targets = self._move_stem_batches(art_targets, device=target_device)
        ref_targets = self._move_stem_batches(ref_targets, device=target_device)

        sep_output = self.sep_loss(
            pred_by_stem=self._merge_stem_batches(art_out.stem_preds, ref_out.stem_preds),
            target_by_stem=self._merge_stem_batches(art_targets, ref_targets),
        )
        sep_loss = sep_output.loss
        asid_output = self.asid_loss(art_out.stem_embeds, ref_out.stem_embeds, pos_masks)
        lambda_asid = self._lambda_asid()
        objective_sep_term = self.lambda_sep * sep_loss
        objective_asid_term = lambda_asid * asid_output.loss
        objective_total = objective_sep_term + objective_asid_term

        is_train = stage == "train"
        is_val = stage == "val"
        batch_size = int(batch.mix_A.shape[0])
        log_kwargs = {
            "on_step": is_train,
            "on_epoch": is_val,
            "sync_dist": is_val,
            "batch_size": batch_size,
        }
        self.log(f"{stage}/loss", objective_total, prog_bar=True, **log_kwargs)
        self.log(f"{stage}/objective_total", objective_total, **log_kwargs)
        self.log(f"{stage}/objective/sep_term", objective_sep_term, **log_kwargs)
        self.log(f"{stage}/objective/asid_term", objective_asid_term, **log_kwargs)
        self.log(f"{stage}/loss_raw/sep", sep_loss, **log_kwargs)
        self.log(f"{stage}/loss_raw/asid", asid_output.loss, **log_kwargs)
        self.log(f"{stage}/lambda_sep", float(self.lambda_sep), **log_kwargs)
        self.log(f"{stage}/lambda_asid", float(lambda_asid), **log_kwargs)
        if is_val:
            self._log_val_stem_metrics(stage=stage, sep_output=sep_output, asid_output=asid_output, device=objective_total.device)
        else:
            self._log_train_stem_metrics(stage=stage, sep_output=sep_output, asid_output=asid_output, log_kwargs=log_kwargs)
        self._log_optimizer_lrs(stage=stage, on_step=is_train, on_epoch=not is_train, batch_size=batch_size)

        return StepOutput(
            loss=objective_total,
            sep_loss=sep_loss,
            asid_loss=asid_output.loss,
            n_anchor=asid_output.n_anchor,
            per_stem_sep_loss=sep_output.per_stem_loss,
            per_stem_asid_loss=asid_output.per_stem_loss,
            per_stem_asid_anchor_count=asid_output.per_stem_anchor_count,
        )

    def training_step(self, batch: SepFPTrainBatch, batch_idx: int) -> torch.Tensor:
        _ = batch_idx
        return self.shared_step(batch, stage="train").loss

    def validation_step(self, batch: SepFPTrainBatch, batch_idx: int) -> torch.Tensor:
        _ = batch_idx
        return self.shared_step(batch, stage="val").loss

    def configure_optimizers(self):
        if self.optimizer_factory is None:
            optimizer = build_sepfp_optimizer(module=self)
        else:
            try:
                optimizer = self.optimizer_factory(module=self)
            except TypeError:
                optimizer = self.optimizer_factory(params=self.parameters())

        if self.scheduler_factory is None:
            return {"optimizer": optimizer}

        try:
            scheduler = self.scheduler_factory(optimizer=optimizer)
        except TypeError:
            scheduler = self.scheduler_factory(optimizer=optimizer, trainer=getattr(self, "trainer", None))

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/objective/asid_term",
            },
        }
