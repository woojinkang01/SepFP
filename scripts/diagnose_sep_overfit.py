from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import hydra
import lightning as L
import rootutils
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from sepfp.data import STEM_ORDER
from sepfp.data.batch_types import BranchContext, SepFPTrainBatch, StemBatch
from sepfp.data.datamodule import sepfp_collate_fn
from sepfp.data.dataset import SepFPDataset
from sepfp.data.effects import RandomizedEffectChain
from sepfp.data.preprocess import build_art_branch, build_ref_branch
from sepfp.data.targets import build_sep_targets
from sepfp.losses.separation import SeparationLoss
from sepfp.models.sepfp_model import SepFPModel
from sepfp.training.module import SepFPLightningModule


@dataclass(frozen=True)
class BranchDiagnostics:
    sep_loss: float
    uniform_loss: float
    ratio_oracle_loss: float
    sum_pred_carrier_l1: float
    sum_target_carrier_l1: float
    carrier_mean: float
    carrier_p95: float
    carrier_max: float
    target_mean: float
    target_p95: float
    target_max: float
    pred_mean: float
    pred_p95: float
    pred_max: float
    mask_entropy: float
    mask_max: float
    active_stems_mean: float


def _move_batch(batch: SepFPTrainBatch, device: torch.device) -> SepFPTrainBatch:
    batch.mix_A = batch.mix_A.to(device)
    batch.mix_B = batch.mix_B.to(device)
    batch.mix_AB = batch.mix_AB.to(device)
    batch.frame_offsets = batch.frame_offsets.to(device)
    for stem_group in (
        batch.individual_stems_A,
        batch.individual_stems_B,
        batch.individual_stems_AB,
    ):
        for source_by_stem in stem_group:
            for sources in source_by_stem.values():
                for source in sources:
                    source.audio = source.audio.to(device)
    return batch


def _quantile(x: torch.Tensor, q: float) -> torch.Tensor:
    return torch.quantile(x.detach().flatten().float().cpu(), q).to(x.device)


def _mean_or_zero(values: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    if not values:
        return torch.zeros((), device=device)
    return torch.stack(values).mean()


def _index_by_sample(batch: StemBatch) -> dict[int, int]:
    return {int(sample_idx.item()): row_idx for row_idx, sample_idx in enumerate(batch.sample_idx)}


def _branch_diagnostics(
    ctx: BranchContext,
    pred_by_stem: dict[str, StemBatch],
    target_by_stem: dict[str, StemBatch],
    sep_loss: SeparationLoss,
    stems: tuple[str, ...],
) -> BranchDiagnostics:
    device = ctx.x_linear_mag.device
    eps = torch.tensor(1e-8, device=device)

    sep = sep_loss(pred_by_stem, target_by_stem).loss.detach()
    uniform_losses: list[torch.Tensor] = []
    oracle_losses: list[torch.Tensor] = []
    pred_carrier_losses: list[torch.Tensor] = []
    target_carrier_losses: list[torch.Tensor] = []
    target_values: list[torch.Tensor] = []
    pred_values: list[torch.Tensor] = []
    entropy_values: list[torch.Tensor] = []
    mask_max_values: list[torch.Tensor] = []

    for sample_idx in range(ctx.x_linear_mag.size(0)):
        active_stems = [stem for stem_idx, stem in enumerate(stems) if bool(ctx.active_mask[sample_idx, stem_idx])]
        if not active_stems:
            continue

        carrier = ctx.x_linear_mag[sample_idx]
        target_maps = []
        pred_maps = []
        mask_maps = []
        for stem in active_stems:
            if stem not in target_by_stem or stem not in pred_by_stem:
                continue
            target_rows = _index_by_sample(target_by_stem[stem])
            pred_rows = _index_by_sample(pred_by_stem[stem])
            if sample_idx not in target_rows or sample_idx not in pred_rows:
                continue

            target = target_by_stem[stem].tensor[target_rows[sample_idx]].to(device)
            pred = pred_by_stem[stem].tensor[pred_rows[sample_idx]]
            mask = pred_by_stem[stem].extras["mask"][pred_rows[sample_idx]].to(device)
            target_maps.append(target)
            pred_maps.append(pred)
            mask_maps.append(mask)
            target_values.append(target.detach())
            pred_values.append(pred.detach())

        if not target_maps:
            continue

        targets = torch.stack(target_maps, dim=0)
        preds = torch.stack(pred_maps, dim=0)
        masks = torch.stack(mask_maps, dim=0)
        target_sum = targets.sum(dim=0)
        pred_sum = preds.sum(dim=0)

        uniform_pred = carrier.unsqueeze(0) / float(len(target_maps))
        target_ratio = targets / (target_sum.unsqueeze(0) + eps)
        oracle_pred = target_ratio * carrier.unsqueeze(0)

        uniform_losses.append(torch.abs(uniform_pred - targets).mean())
        oracle_losses.append(torch.abs(oracle_pred - targets).mean())
        pred_carrier_losses.append(torch.abs(pred_sum - carrier).mean())
        target_carrier_losses.append(torch.abs(target_sum - carrier).mean())

        entropy = -(masks.clamp_min(float(eps)).log() * masks).sum(dim=0)
        entropy_values.append(entropy.mean())
        mask_max_values.append(masks.max(dim=0).values.mean())

    carrier = ctx.x_linear_mag.detach()
    target_cat = torch.cat([x.flatten() for x in target_values]) if target_values else torch.zeros(1, device=device)
    pred_cat = torch.cat([x.flatten() for x in pred_values]) if pred_values else torch.zeros(1, device=device)

    return BranchDiagnostics(
        sep_loss=float(sep.item()),
        uniform_loss=float(_mean_or_zero(uniform_losses, device).item()),
        ratio_oracle_loss=float(_mean_or_zero(oracle_losses, device).item()),
        sum_pred_carrier_l1=float(_mean_or_zero(pred_carrier_losses, device).item()),
        sum_target_carrier_l1=float(_mean_or_zero(target_carrier_losses, device).item()),
        carrier_mean=float(carrier.mean().item()),
        carrier_p95=float(_quantile(carrier, 0.95).item()),
        carrier_max=float(carrier.max().item()),
        target_mean=float(target_cat.mean().item()),
        target_p95=float(_quantile(target_cat, 0.95).item()),
        target_max=float(target_cat.max().item()),
        pred_mean=float(pred_cat.mean().item()),
        pred_p95=float(_quantile(pred_cat, 0.95).item()),
        pred_max=float(pred_cat.max().item()),
        mask_entropy=float(_mean_or_zero(entropy_values, device).item()),
        mask_max=float(_mean_or_zero(mask_max_values, device).item()),
        active_stems_mean=float(ctx.active_mask.sum(dim=1).float().mean().item()),
    )


def _merge_stem_batches(first: dict[str, StemBatch], second: dict[str, StemBatch]) -> dict[str, StemBatch]:
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
            extras={},
        )
    return merged


def _sep_step(
    module: SepFPLightningModule,
    batch: SepFPTrainBatch,
    effect_chain: RandomizedEffectChain,
) -> tuple[torch.Tensor, dict[str, BranchDiagnostics]]:
    with torch.no_grad():
        x_A_complex = module.transform(batch.mix_A)
        x_B_complex = module.transform(batch.mix_B)
        x_AB_complex = module.transform(batch.mix_AB)
        art_ctx = build_art_branch(
            batch=batch,
            x_A_complex=x_A_complex,
            x_B_complex=x_B_complex,
            block_size=module.block_size,
            mean=module.lognorm_mean,
            std=module.lognorm_std,
            pitch_shift=module.pitch_shift,
            crop_size=module._crop_size(),
            stems=module.stems,
        )
        ref_ctx = build_ref_branch(
            batch=batch,
            x_AB_complex=x_AB_complex,
            block_size=module.block_size,
            mean=module.lognorm_mean,
            std=module.lognorm_std,
            crop_size=module._crop_size(),
            time_stretch=module.time_stretch,
            stems=module.stems,
        )

    art_out = module.model.forward_branch(art_ctx)
    ref_out = module.model.forward_branch(ref_ctx)

    with torch.no_grad():
        art_targets, ref_targets = build_sep_targets(
            batch=batch,
            art_ctx=art_ctx,
            ref_ctx=ref_ctx,
            vqt_transform=module.transform,
            apply_effects=effect_chain.apply_with_params,
            sample_rate=module.sample_rate,
            block_size=module.block_size,
            mean=module.lognorm_mean,
            std=module.lognorm_std,
            stems=module.stems,
        )

    total_loss = module.sep_loss(
        pred_by_stem=_merge_stem_batches(art_out.stem_preds, ref_out.stem_preds),
        target_by_stem=_merge_stem_batches(art_targets, ref_targets),
    ).loss
    diagnostics = {
        "art": _branch_diagnostics(art_ctx, art_out.stem_preds, art_targets, module.sep_loss, module.stems),
        "ref": _branch_diagnostics(ref_ctx, ref_out.stem_preds, ref_targets, module.sep_loss, module.stems),
    }
    return total_loss, diagnostics


def _grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    norms = [p.grad.detach().norm(2) for p in parameters if p.grad is not None]
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2).item())


def _build_module(cfg, device: torch.device) -> tuple[SepFPLightningModule, RandomizedEffectChain]:
    effect_chain = RandomizedEffectChain(cfg.data.dataset.board)
    model = SepFPModel(
        stems=tuple(cfg.model.stems),
        encoder_channels=cfg.model.encoder.out_channels,
        evidence_channels=cfg.model.evidence.channels,
        query_dim=cfg.model.evidence.query_dim,
        attention_heads=cfg.model.evidence.num_attention_heads,
        decoder_hidden_channels=cfg.model.decoder.hidden_channels,
        projector_hidden_channels=cfg.model.projector.hidden_channels,
        projector_out_dim=cfg.model.projector.out_dim,
        mask_mode=cfg.model.decoder.get("mask_mode", cfg.model.decoder.get("mask_normalization", "active_softmax")),
        max_mask=cfg.model.decoder.get("max_mask", 2.0),
    )
    module = SepFPLightningModule(
        model=model,
        transform=hydra.utils.instantiate(cfg.model.transform),
        sep_loss=SeparationLoss(),
        sample_rate=cfg.data.dataset.sample_rate,
        block_size=tuple(cfg.model.block_size),
        time_stretch=tuple(cfg.model.time_stretch) if cfg.model.get("time_stretch") is not None else None,
        pitch_shift=cfg.model.pitch_shift,
        lognorm_mean=cfg.data.norm_stats[0],
        lognorm_std=cfg.data.norm_stats[1],
        stems=tuple(cfg.model.stems),
        lambda_sep=1.0,
        lambda_asid_final=0.0,
        lambda_asid_warmup_epochs=0,
        apply_effects=effect_chain.apply_with_params,
    )
    return module.to(device), effect_chain


def _build_loader(cfg, batch_size: int, max_examples: int, shuffle: bool) -> DataLoader:
    dataset = SepFPDataset(**OmegaConf.to_container(cfg.data.dataset, resolve=True))
    if max_examples > 0:
        dataset = Subset(dataset, list(range(min(max_examples, len(dataset)))))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=sepfp_collate_fn)


def _write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run separation-only SepFP diagnostics on real batches.")
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--config-path", default="../configs")
    parser.add_argument("--data", default="full")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--mode", choices=("one-batch", "tiny-subset"), default="one-batch")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="outputs/sep_diagnostics/diagnostics.jsonl")
    parser.add_argument("--fixed-augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--mask-mode",
        choices=("active_softmax", "independent_sigmoid", "independent_softplus", "independent_capped"),
        default="active_softmax",
    )
    parser.add_argument("--max-mask", type=float, default=2.0)
    args = parser.parse_args()

    L.seed_everything(args.seed, workers=True)
    overrides = [
        f"data={args.data}",
        "loss.asid.lambda_final=0.0",
        "loss.asid.warmup_epochs=0",
        f"model.decoder.mask_mode={args.mask_mode}",
        f"model.decoder.max_mask={args.max_mask}",
    ]
    with hydra.initialize(config_path=args.config_path, version_base="1.3"):
        cfg = hydra.compose(config_name=args.config_name, overrides=overrides)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    module, effect_chain = _build_module(cfg, device)
    module.train()
    optimizer = torch.optim.AdamW(module.model.parameters(), lr=args.lr)
    loader = _build_loader(
        cfg=cfg,
        batch_size=args.batch_size,
        max_examples=args.max_examples,
        shuffle=args.mode == "tiny-subset",
    )
    fixed_batch = _move_batch(next(iter(loader)), device)
    tiny_batches = [_move_batch(batch, device) for batch in loader] if args.mode == "tiny-subset" else [fixed_batch]
    output_path = Path(args.output)
    if output_path.exists():
        output_path.unlink()

    for step in range(args.steps + 1):
        batch = fixed_batch if args.mode == "one-batch" else tiny_batches[step % len(tiny_batches)]
        if args.fixed_augment:
            torch.manual_seed(args.seed + (step % len(tiny_batches) if args.mode == "tiny-subset" else 0))
        optimizer.zero_grad(set_to_none=True)
        loss, diagnostics = _sep_step(module, batch, effect_chain)
        if step > 0:
            loss.backward()
            grad_norm = _grad_norm(module.model.parameters())
            optimizer.step()
        else:
            grad_norm = 0.0

        if step == 0 or step == args.steps or step % args.log_every == 0:
            record = {
                "step": step,
                "mode": args.mode,
                "mask_mode": args.mask_mode,
                "max_mask": args.max_mask,
                "device": str(device),
                "loss": float(loss.detach().item()),
                "grad_norm": grad_norm,
                "art": asdict(diagnostics["art"]),
                "ref": asdict(diagnostics["ref"]),
            }
            print(json.dumps(record, sort_keys=True))
            _write_record(output_path, record)


if __name__ == "__main__":
    main()
