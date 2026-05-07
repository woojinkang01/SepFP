from __future__ import annotations

from typing import Callable

import torch

from sepfp.data.batch_types import BranchContext, STEM_ORDER, SepFPTrainBatch, StemBatch, StemSource
from sepfp.data.preprocess import complex_to_linear_mag, tracked_extract_random_blocks, tracked_stretch_and_crop


def _apply_effect_to_sources(
    sources: tuple[StemSource, ...],
    sample_rate: int,
    effect_params,
    apply_effects: Callable[[torch.Tensor, int, object], torch.Tensor] | None,
    device: torch.device,
) -> list[torch.Tensor]:
    if not effect_params.ops:
        outputs = []
        for source in sources:
            audio = source.audio.mean(dim=0) if source.audio.ndim > 1 else source.audio
            outputs.append(audio.to(device=device))
        return outputs
    if apply_effects is None:
        raise RuntimeError("Effect parameters are present but no effect applier was provided")
    outputs = []
    for source in sources:
        effected = apply_effects(source.audio, sample_rate, effect_params)
        outputs.append(effected.to(device=device))
    return outputs


def _complex_sum(components: list[torch.Tensor]) -> torch.Tensor:
    if not components:
        raise ValueError("Cannot sum an empty complex target list")
    total = components[0]
    for component in components[1:]:
        total = total + component
    return total


def build_sep_targets(
    batch: SepFPTrainBatch,
    art_ctx: BranchContext,
    ref_ctx: BranchContext,
    vqt_transform: Callable[[torch.Tensor], torch.Tensor],
    apply_effects: Callable[[torch.Tensor, int, object], torch.Tensor],
    sample_rate: int,
    block_size: tuple[int, int],
    mean: float,
    std: float,
    stems: tuple[str, ...] = STEM_ORDER,
) -> tuple[dict[str, StemBatch], dict[str, StemBatch]]:
    art_targets: dict[str, StemBatch] = {}
    ref_targets: dict[str, StemBatch] = {}
    batch_size = batch.mix_A.size(0)
    device = batch.mix_AB.device

    for stem_idx, stem in enumerate(stems):
        art_sample_idx = torch.nonzero(art_ctx.active_mask[:, stem_idx], as_tuple=False).flatten()
        ref_sample_idx = torch.nonzero(ref_ctx.active_mask[:, stem_idx], as_tuple=False).flatten()

        if art_sample_idx.numel() > 0:
            targets = []
            provenance_rows = []
            for sample_idx in art_sample_idx.tolist():
                b_src = int(art_ctx.crop_meta["rolled_from"][sample_idx].item())
                components = []

                for audio in _apply_effect_to_sources(
                    batch.individual_stems_A[sample_idx].get(stem, ()),
                    sample_rate,
                    batch.effect_params_A[sample_idx],
                    apply_effects,
                    device,
                ):
                    complex_spec = vqt_transform(audio.unsqueeze(0)).squeeze(0)
                    crop, _ = tracked_extract_random_blocks(
                        complex_spec.unsqueeze(0),
                        block_size,
                        i=int(art_ctx.crop_meta["i_A"][sample_idx].item()),
                        j=int(art_ctx.crop_meta["j_A"][sample_idx].item()),
                    )
                    components.append(crop.squeeze(0))

                for audio in _apply_effect_to_sources(
                    batch.individual_stems_B[b_src].get(stem, ()),
                    sample_rate,
                    batch.effect_params_B[b_src],
                    apply_effects,
                    device,
                ):
                    complex_spec = vqt_transform(audio.unsqueeze(0)).squeeze(0)
                    crop, _ = tracked_extract_random_blocks(
                        complex_spec.unsqueeze(0),
                        block_size,
                        i=int(art_ctx.crop_meta["i_B"][b_src].item()),
                        j=int(art_ctx.crop_meta["j_B"][b_src].item()),
                    )
                    components.append(crop.squeeze(0))

                target_complex = _complex_sum(components)
                target_linear_mag = complex_to_linear_mag(target_complex.unsqueeze(0)).squeeze(0)
                targets.append(target_linear_mag)
                provenance_rows.append(tuple(art_ctx.provenance[sample_idx].get(stem, ())))

            art_targets[stem] = StemBatch(
                sample_idx=art_sample_idx.to(dtype=torch.long),
                tensor=torch.stack(targets, dim=0),
                provenance=tuple(provenance_rows),
            )

        if ref_sample_idx.numel() > 0:
            targets = []
            provenance_rows = []
            for sample_idx in ref_sample_idx.tolist():
                components = []
                stretch = torch.tensor([float(ref_ctx.crop_meta["stretch"][sample_idx].item())], device=batch.mix_AB.device)
                for audio in _apply_effect_to_sources(
                    batch.individual_stems_AB[sample_idx].get(stem, ()),
                    sample_rate,
                    batch.effect_params_AB[sample_idx],
                    apply_effects,
                    device,
                ):
                    complex_spec = vqt_transform(audio.unsqueeze(0)).squeeze(0)
                    crop, _ = tracked_stretch_and_crop(
                        complex_spec.unsqueeze(0),
                        block_size=block_size,
                        stretch_factor=(1.0, 1.0),
                        i=int(ref_ctx.crop_meta["i"][sample_idx].item()),
                        j=int(ref_ctx.crop_meta["j"][sample_idx].item()),
                        s=stretch,
                        pad_left=ref_ctx.crop_meta["pad_left"][sample_idx],
                    )
                    components.append(crop.squeeze(0))

                target_complex = _complex_sum(components)
                target_linear_mag = complex_to_linear_mag(target_complex.unsqueeze(0)).squeeze(0)
                targets.append(target_linear_mag)
                provenance_rows.append(tuple(ref_ctx.provenance[sample_idx].get(stem, ())))

            ref_targets[stem] = StemBatch(
                sample_idx=ref_sample_idx.to(dtype=torch.long),
                tensor=torch.stack(targets, dim=0),
                provenance=tuple(provenance_rows),
            )

    return art_targets, ref_targets
