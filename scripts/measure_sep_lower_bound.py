from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import rootutils
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from sepfp.data.batch_types import BranchContext, SepFPTrainBatch, StemBatch
from sepfp.data.datamodule import sepfp_collate_fn
from sepfp.data.dataset import SepFPDataset
from sepfp.data.effects import RandomizedEffectChain
from sepfp.data.preprocess import build_art_branch, build_ref_branch
from sepfp.data.targets import build_sep_targets


@dataclass
class MeanStat:
    total: float = 0.0
    count: float = 0.0

    def add(self, value: float, count: float = 1.0) -> None:
        self.total += float(value) * float(count)
        self.count += float(count)

    def add_total(self, total: float, count: float) -> None:
        self.total += float(total)
        self.count += float(count)

    def value(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


@dataclass
class BranchAccumulator:
    name: str
    stats: dict[str, MeanStat] = field(default_factory=lambda: defaultdict(MeanStat))
    per_stem: dict[str, dict[str, MeanStat]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(MeanStat)))
    by_active_count: dict[int, dict[str, MeanStat]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(MeanStat)))
    max_values: dict[str, float] = field(default_factory=dict)

    def add_stat(self, name: str, value: float, count: float = 1.0) -> None:
        self.stats[name].add(value, count=count)

    def add_total(self, name: str, total: float, count: float) -> None:
        self.stats[name].add_total(total=total, count=count)

    def add_stem_stat(self, stem: str, name: str, value: float, count: float = 1.0) -> None:
        self.per_stem[stem][name].add(value, count=count)

    def add_active_count_stat(self, active_count: int, name: str, value: float, count: float = 1.0) -> None:
        self.by_active_count[active_count][name].add(value, count=count)

    def add_active_count_total(self, active_count: int, name: str, total: float, count: float) -> None:
        self.by_active_count[active_count][name].add_total(total=total, count=count)

    def set_max(self, name: str, value: float) -> None:
        self.max_values[name] = max(self.max_values.get(name, float("-inf")), float(value))

    def snapshot(self) -> dict[str, Any]:
        metrics = {name: stat.value() for name, stat in sorted(self.stats.items())}
        counts = {
            "samples": int(self.stats["samples"].total),
            "covered_samples": int(self.stats["covered_samples"].total),
            "active_instances": int(self.stats["active_instances"].total),
        }
        per_stem = {
            stem: {
                name: int(stat.total) if name == "count" else stat.value()
                for name, stat in sorted(stem_stats.items())
            }
            for stem, stem_stats in sorted(self.per_stem.items())
        }
        by_active_count = {
            str(active_count): {
                name: int(stat.total) if name == "samples" else stat.value()
                for name, stat in sorted(count_stats.items())
            }
            for active_count, count_stats in sorted(self.by_active_count.items())
        }
        return {
            "name": self.name,
            "counts": counts,
            "metrics": metrics,
            "per_stem": per_stem,
            "by_active_count": by_active_count,
            "max_values": dict(sorted(self.max_values.items())),
        }


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


def _index_by_sample(batch: StemBatch) -> dict[int, int]:
    return {int(sample_idx.item()): row_idx for row_idx, sample_idx in enumerate(batch.sample_idx)}


def _quantile_or_zero(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.detach().flatten().float().cpu(), q).item())


def _mean_or_zero(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.detach().float().mean().item())


def _measure_branch(
    ctx: BranchContext,
    target_by_stem: dict[str, StemBatch],
    stems: tuple[str, ...],
    max_mask: float,
    eps: float,
) -> dict[str, Any]:
    batch_acc = BranchAccumulator(name=ctx.name)
    batch_size = int(ctx.x_linear_mag.size(0))
    device = ctx.x_linear_mag.device
    target_rows_by_stem = {stem: _index_by_sample(batch) for stem, batch in target_by_stem.items()}
    target_values: list[torch.Tensor] = []
    carrier_values: list[torch.Tensor] = []
    ratio_values: list[torch.Tensor] = []

    batch_acc.add_stat("samples", batch_size, count=1)

    for sample_idx in range(batch_size):
        active_targets: list[tuple[str, torch.Tensor]] = []
        for stem_idx, stem in enumerate(stems):
            if not bool(ctx.active_mask[sample_idx, stem_idx]):
                continue
            rows = target_rows_by_stem.get(stem)
            if rows is None or sample_idx not in rows:
                continue
            target = target_by_stem[stem].tensor[rows[sample_idx]].to(device)
            active_targets.append((stem, target))

        active_count = len(active_targets)
        if active_count == 0:
            continue

        carrier = ctx.x_linear_mag[sample_idx]
        targets = torch.stack([target for _, target in active_targets], dim=0)
        target_sum = targets.sum(dim=0)
        uniform_pred = carrier.unsqueeze(0) / float(active_count)
        ratio_oracle_pred = targets / (target_sum.unsqueeze(0) + eps) * carrier.unsqueeze(0)
        capped_oracle_pred = torch.minimum(targets, carrier.unsqueeze(0) * float(max_mask))

        independent_losses = torch.abs(capped_oracle_pred - targets).mean(dim=(1, 2, 3))
        ratio_oracle_losses = torch.abs(ratio_oracle_pred - targets).mean(dim=(1, 2, 3))
        uniform_losses = torch.abs(uniform_pred - targets).mean(dim=(1, 2, 3))
        zero_losses = targets.mean(dim=(1, 2, 3))
        unit_losses = torch.abs(carrier.unsqueeze(0) - targets).mean(dim=(1, 2, 3))
        active_softmax_total = torch.abs(target_sum - carrier).mean()
        sum_target_carrier_l1 = active_softmax_total

        batch_acc.add_stat("covered_samples", 1, count=1)
        batch_acc.add_stat("active_instances", active_count, count=1)
        batch_acc.add_stat("active_stems_mean", active_count, count=1)
        batch_acc.add_total("lb_independent_capped", float(independent_losses.sum().item()), active_count)
        batch_acc.add_total("ratio_oracle_loss", float(ratio_oracle_losses.sum().item()), active_count)
        batch_acc.add_total("uniform_loss", float(uniform_losses.sum().item()), active_count)
        batch_acc.add_total("zero_mask_loss", float(zero_losses.sum().item()), active_count)
        batch_acc.add_total("unit_mask_loss", float(unit_losses.sum().item()), active_count)
        batch_acc.add_total("lb_active_softmax_exact", float(active_softmax_total.item()), active_count)
        batch_acc.add_stat("sum_target_carrier_l1", float(sum_target_carrier_l1.item()), count=1)
        batch_acc.add_active_count_total(
            active_count,
            "lb_independent_capped",
            float(independent_losses.sum().item()),
            active_count,
        )
        batch_acc.add_active_count_total(
            active_count,
            "lb_active_softmax_exact",
            float(active_softmax_total.item()),
            active_count,
        )
        batch_acc.add_active_count_stat(active_count, "sum_target_carrier_l1", float(sum_target_carrier_l1.item()))
        batch_acc.add_active_count_stat(active_count, "samples", 1)

        for local_idx, (stem, target) in enumerate(active_targets):
            batch_acc.add_stem_stat(stem, "count", 1)
            batch_acc.add_stem_stat(stem, "lb_independent_capped", float(independent_losses[local_idx].item()))
            batch_acc.add_stem_stat(stem, "ratio_oracle_loss", float(ratio_oracle_losses[local_idx].item()))
            batch_acc.add_stem_stat(stem, "uniform_loss", float(uniform_losses[local_idx].item()))
            batch_acc.add_stem_stat(stem, "zero_mask_loss", float(zero_losses[local_idx].item()))
            batch_acc.add_stem_stat(stem, "unit_mask_loss", float(unit_losses[local_idx].item()))
            batch_acc.add_stem_stat(stem, "target_mean", float(target.mean().item()))

        target_values.append(targets.detach().flatten())
        carrier_values.append(carrier.detach().flatten())
        ratio_values.append((targets / (carrier.unsqueeze(0) + eps)).detach().flatten())

    target_flat = torch.cat(target_values) if target_values else torch.empty(0, device=device)
    carrier_flat = torch.cat(carrier_values) if carrier_values else torch.empty(0, device=device)
    ratio_flat = torch.cat(ratio_values) if ratio_values else torch.empty(0, device=device)

    for name, values in (
        ("target", target_flat),
        ("carrier", carrier_flat),
        ("target_carrier_ratio", ratio_flat),
    ):
        batch_acc.add_stat(f"{name}_mean", _mean_or_zero(values))
        batch_acc.add_stat(f"{name}_p50", _quantile_or_zero(values, 0.50))
        batch_acc.add_stat(f"{name}_p95", _quantile_or_zero(values, 0.95))
        batch_acc.add_stat(f"{name}_p99", _quantile_or_zero(values, 0.99))
        max_value = float(values.detach().max().item()) if values.numel() > 0 else 0.0
        batch_acc.set_max(f"{name}_max", max_value)

    return batch_acc.snapshot()


def _merge_summary(target: BranchAccumulator, batch_record: dict[str, Any]) -> None:
    counts = batch_record["counts"]
    metrics = batch_record["metrics"]
    active_instances = max(counts["active_instances"], 1)
    covered_samples = max(counts["covered_samples"], 1)

    for count_name, value in counts.items():
        target.add_stat(count_name, value, count=1)

    instance_weighted = {
        "lb_independent_capped",
        "lb_active_softmax_exact",
        "ratio_oracle_loss",
        "uniform_loss",
        "zero_mask_loss",
        "unit_mask_loss",
    }
    sample_weighted = {"active_stems_mean", "sum_target_carrier_l1"}
    for name, value in metrics.items():
        if name in instance_weighted:
            target.add_stat(name, value, count=active_instances)
        elif name in sample_weighted:
            target.add_stat(name, value, count=covered_samples)
        else:
            target.add_stat(f"mean_batch_{name}", value, count=1)

    for stem, stem_metrics in batch_record["per_stem"].items():
        stem_count = max(int(stem_metrics.get("count", 0.0)), 1)
        for name, value in stem_metrics.items():
            if name == "count":
                target.add_stem_stat(stem, name, value, count=1)
            else:
                target.add_stem_stat(stem, name, value, count=stem_count)

    for active_count, active_metrics in batch_record["by_active_count"].items():
        active_count_int = int(active_count)
        sample_count = max(int(active_metrics.get("samples", 0.0)), 1)
        instance_count = active_count_int * sample_count
        for name, value in active_metrics.items():
            if name == "samples":
                target.add_active_count_stat(active_count_int, name, value, count=1)
            elif name in instance_weighted:
                target.add_active_count_stat(active_count_int, name, value, count=instance_count)
            else:
                target.add_active_count_stat(active_count_int, name, value, count=sample_count)

    for name, value in batch_record["max_values"].items():
        target.set_max(name, value)


def _build_loader(dataset_cfg: Any, batch_size: int, max_examples: int) -> DataLoader:
    dataset = SepFPDataset(**OmegaConf.to_container(dataset_cfg, resolve=True))
    if max_examples > 0:
        dataset = Subset(dataset, list(range(min(max_examples, len(dataset)))))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=sepfp_collate_fn)


def _select_dataset_cfg(cfg: Any, split: str) -> Any:
    if split == "train":
        return cfg.data.dataset
    if split == "val":
        if cfg.data.get("validation_dataset") is None:
            raise ValueError("Selected split=val, but cfg.data.validation_dataset is not configured")
        return cfg.data.validation_dataset
    raise ValueError(f"Unknown split: {split}")


def _write_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _measure_split(
    cfg: Any,
    split: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    L.seed_everything(seed, workers=True)
    dataset_cfg = _select_dataset_cfg(cfg, split)
    loader = _build_loader(dataset_cfg, batch_size=args.batch_size, max_examples=args.max_examples)
    transform = hydra.utils.instantiate(cfg.model.transform).to(device)
    effect_chain = RandomizedEffectChain(dataset_cfg.get("board", ()))
    stems = tuple(cfg.model.stems)
    block_size = tuple(cfg.model.block_size)
    time_stretch = tuple(cfg.model.time_stretch) if cfg.model.get("time_stretch") is not None else None
    max_mask = float(args.max_mask if args.max_mask is not None else cfg.model.decoder.get("max_mask", 2.0))

    summary = {
        "art": BranchAccumulator(name="art"),
        "ref": BranchAccumulator(name="ref"),
    }
    output_path = Path(args.output)

    for batch_idx, batch in enumerate(loader):
        if args.num_batches > 0 and batch_idx >= args.num_batches:
            break
        batch = _move_batch(batch, device)
        with torch.no_grad():
            x_A_complex = transform(batch.mix_A)
            x_B_complex = transform(batch.mix_B)
            x_AB_complex = transform(batch.mix_AB)
            art_ctx = build_art_branch(
                batch=batch,
                x_A_complex=x_A_complex,
                x_B_complex=x_B_complex,
                block_size=block_size,
                mean=cfg.data.norm_stats[0],
                std=cfg.data.norm_stats[1],
                pitch_shift=cfg.model.pitch_shift,
                crop_size=args.pitch_crop_bins,
                stems=stems,
            )
            ref_ctx = build_ref_branch(
                batch=batch,
                x_AB_complex=x_AB_complex,
                block_size=block_size,
                mean=cfg.data.norm_stats[0],
                std=cfg.data.norm_stats[1],
                crop_size=args.pitch_crop_bins,
                time_stretch=time_stretch,
                stems=stems,
            )
            art_targets, ref_targets = build_sep_targets(
                batch=batch,
                art_ctx=art_ctx,
                ref_ctx=ref_ctx,
                vqt_transform=transform,
                apply_effects=effect_chain.apply_with_params,
                sample_rate=dataset_cfg.sample_rate,
                block_size=block_size,
                mean=cfg.data.norm_stats[0],
                std=cfg.data.norm_stats[1],
                stems=stems,
            )

            branches = {
                "art": _measure_branch(
                    ctx=art_ctx,
                    target_by_stem=art_targets,
                    stems=stems,
                    max_mask=max_mask,
                    eps=args.eps,
                ),
                "ref": _measure_branch(
                    ctx=ref_ctx,
                    target_by_stem=ref_targets,
                    stems=stems,
                    max_mask=max_mask,
                    eps=args.eps,
                ),
            }

        for branch_name, branch_record in branches.items():
            _merge_summary(summary[branch_name], branch_record)

        record = {
            "type": "batch",
            "seed": seed,
            "split": split,
            "batch_idx": batch_idx,
            "batch_size": int(batch.mix_A.size(0)),
            "max_mask": max_mask,
            "branches": branches,
        }
        print(json.dumps(record, sort_keys=True))
        _write_record(output_path, record)

    summary_record = {
        "type": "summary",
        "seed": seed,
        "split": split,
        "max_mask": max_mask,
        "branches": {branch_name: acc.snapshot() for branch_name, acc in summary.items()},
    }
    print(json.dumps(summary_record, sort_keys=True))
    _write_record(output_path, summary_record)
    return summary_record


def _parse_seeds(raw: str) -> list[int]:
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure SepFP separation target/carrier oracle lower bounds.")
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--config-path", default="../configs")
    parser.add_argument("--data", default="full")
    parser.add_argument("--split", choices=("train", "val", "both"), default="val")
    parser.add_argument("--seeds", default="0", help="Comma-separated seeds, e.g. '0,1,2'.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=0, help="0 means scan the selected split.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="outputs/sep_lower_bound/lower_bound.jsonl")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--max-mask", type=float, default=None)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--pitch-crop-bins", type=int, default=18)
    args = parser.parse_args()

    overrides = [f"data={args.data}"]
    with hydra.initialize(config_path=args.config_path, version_base="1.3"):
        cfg = hydra.compose(config_name=args.config_name, overrides=overrides)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_path = Path(args.output)
    if output_path.exists() and not args.append:
        output_path.unlink()

    splits = ("train", "val") if args.split == "both" else (args.split,)
    run_summaries = []
    for seed in _parse_seeds(args.seeds):
        for split in splits:
            run_summaries.append(_measure_split(cfg=cfg, split=split, seed=seed, args=args, device=device))

    final_record = {
        "type": "run_complete",
        "config": {
            "data": args.data,
            "split": args.split,
            "seeds": _parse_seeds(args.seeds),
            "batch_size": args.batch_size,
            "max_examples": args.max_examples,
            "num_batches": args.num_batches,
            "device": str(device),
        },
        "summaries": run_summaries,
    }
    print(json.dumps(final_record, sort_keys=True))
    _write_record(output_path, final_record)


if __name__ == "__main__":
    main()
