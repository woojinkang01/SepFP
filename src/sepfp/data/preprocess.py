from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F

from sepfp.data.batch_types import BranchContext, STEM_ORDER, SepFPTrainBatch


def build_active_mask(
    stem_types: Iterable[tuple[str, ...]],
    stems: tuple[str, ...] = STEM_ORDER,
) -> torch.BoolTensor:
    rows = []
    for active in stem_types:
        active_set = set(active)
        rows.append([stem in active_set for stem in stems])
    return torch.tensor(rows, dtype=torch.bool)


def generate_indices(
    max_i: int,
    batch_size: int,
    i: torch.Tensor | int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    device = device or torch.device("cpu")
    if torch.is_tensor(i):
        return i.to(device=device, dtype=torch.long)
    if isinstance(i, int):
        return torch.full((batch_size,), i, dtype=torch.long, device=device)
    return torch.randint(0, max_i, (batch_size,), dtype=torch.long, device=device)


def tracked_extract_random_blocks(
    x: torch.Tensor,
    block_size: tuple[int, int],
    i: torch.Tensor | int | None = None,
    j: torch.Tensor | int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch, height, width, _ = x.shape
    block_h, block_w = block_size
    max_i = height - block_h + 1
    max_j = width - block_w + 1
    i_idx = generate_indices(max_i, batch, i=i, device=x.device)
    j_idx = generate_indices(max_j, batch, i=j, device=x.device)

    crops = []
    for batch_idx in range(batch):
        crops.append(x[batch_idx, i_idx[batch_idx] : i_idx[batch_idx] + block_h, j_idx[batch_idx] : j_idx[batch_idx] + block_w, :])

    return torch.stack(crops, dim=0), {"i": i_idx, "j": j_idx}


def tracked_stretch_and_crop(
    x: torch.Tensor,
    block_size: tuple[int, int],
    stretch_factor: tuple[float, float],
    i: torch.Tensor | int | None = None,
    j: torch.Tensor | int | None = None,
    s: torch.Tensor | None = None,
    random_padding: bool = True,
    pad_left: torch.Tensor | int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch = x.size(0)
    block_h, block_w = block_size
    factors = s if s is not None else stretch_factor[0] + torch.rand(batch, device=x.device) * (stretch_factor[1] - stretch_factor[0])

    out = []
    j_indices = []
    i_indices = []
    pad_left_values = []
    pad_right_values = []
    for batch_idx, elem in enumerate(x.permute(0, 3, 1, 2)):
        _, height, width = elem.shape
        new_width = int(width / float(factors[batch_idx]))
        stretched = F.interpolate(elem.unsqueeze(0), size=(height, new_width), mode="bilinear", align_corners=False).squeeze(0)

        if new_width < block_w:
            total_padding = block_w - new_width
            if torch.is_tensor(pad_left):
                left_padding = int(pad_left[batch_idx].item() if pad_left.ndim > 0 else pad_left.item())
            elif isinstance(pad_left, int):
                left_padding = pad_left
            else:
                left_padding = int(torch.randint(total_padding + 1, (1,), device=x.device).item()) if random_padding else 0
            left_padding = max(0, min(left_padding, total_padding))
            right_padding = total_padding - left_padding
            stretched = F.pad(stretched, (left_padding, right_padding))
            new_width = block_w
        else:
            left_padding = 0
            right_padding = 0

        offset_i = generate_indices(height - block_h + 1, 1, i=None if i is None or torch.is_tensor(i) else i, device=x.device)[0]
        if torch.is_tensor(i):
            offset_i = i[batch_idx]

        max_w = new_width - block_w
        if torch.is_tensor(j):
            offset_j = j[batch_idx] if j[batch_idx] <= max_w else torch.tensor(max_w, device=x.device)
        elif isinstance(j, int):
            offset_j = torch.tensor(min(j, max_w), device=x.device)
        else:
            offset_j = torch.randint(max_w + 1, (1,), device=x.device)[0]

        out.append(stretched[:, offset_i : offset_i + block_h, offset_j : offset_j + block_w])
        i_indices.append(offset_i)
        j_indices.append(offset_j)
        pad_left_values.append(torch.tensor(left_padding, dtype=torch.long, device=x.device))
        pad_right_values.append(torch.tensor(right_padding, dtype=torch.long, device=x.device))

    stacked = torch.stack(out, dim=0).permute(0, 2, 3, 1)
    return stacked, {
        "i": torch.stack(i_indices).to(dtype=torch.long),
        "j": torch.stack(j_indices).to(dtype=torch.long),
        "stretch": factors.to(dtype=torch.float32),
        "pad_left": torch.stack(pad_left_values).to(dtype=torch.long),
        "pad_right": torch.stack(pad_right_values).to(dtype=torch.long),
    }


def normalize_logmag_with_gain(
    x_complex: torch.Tensor,
    mean: float,
    std: float,
    gain: torch.Tensor | None = None,
    eps: float = 1e-12,
    return_gain: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    magnitude = x_complex.pow(2).sum(dim=-1).sqrt()
    if gain is None:
        gain = magnitude.max().detach()
    normalized = torch.log(magnitude / (gain + eps) + eps)
    normalized = (normalized - mean) / std
    normalized = normalized.unsqueeze(1)
    if return_gain:
        return normalized, gain
    return normalized


def complex_to_linear_mag(x_complex: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    magnitude = x_complex.pow(2).sum(dim=-1).add(eps).sqrt()
    return magnitude.unsqueeze(1)


def _merge_active_masks(
    stems_a: tuple[tuple[str, ...], ...],
    stems_b: tuple[tuple[str, ...], ...],
    stems: tuple[str, ...],
) -> torch.BoolTensor:
    batch = len(stems_a)
    rows = []
    for batch_idx in range(batch):
        union = set(stems_a[batch_idx]) | set(stems_b[batch_idx])
        rows.append([stem in union for stem in stems])
    return torch.tensor(rows, dtype=torch.bool)


def _merge_provenance(
    provenance_a: tuple[dict[str, tuple[str, ...]], ...],
    provenance_b: tuple[dict[str, tuple[str, ...]], ...],
) -> tuple[dict[str, tuple[str, ...]], ...]:
    merged = []
    for prov_a, prov_b in zip(provenance_a, provenance_b):
        sample: dict[str, tuple[str, ...]] = {}
        for stem in set(prov_a) | set(prov_b):
            sample[stem] = tuple(prov_a.get(stem, ())) + tuple(prov_b.get(stem, ()))
        merged.append(sample)
    return tuple(merged)


def build_art_branch(
    batch: SepFPTrainBatch,
    x_A_complex: torch.Tensor,
    x_B_complex: torch.Tensor,
    block_size: tuple[int, int],
    mean: float,
    std: float,
    pitch_shift: bool,
    crop_size: int,
    stems: tuple[str, ...] = STEM_ORDER,
) -> BranchContext:
    pitch_i = None if pitch_shift else crop_size
    x_A_crop, meta_A = tracked_extract_random_blocks(x_A_complex, block_size, i=pitch_i)
    x_B_crop, meta_B = tracked_extract_random_blocks(x_B_complex, block_size, i=pitch_i)

    x_B_roll = x_B_crop.roll(1, dims=0)
    x_art_complex = x_A_crop + x_B_roll
    x_art_input, gain_art = normalize_logmag_with_gain(x_art_complex, mean=mean, std=std, return_gain=True)
    x_art_linear_mag = complex_to_linear_mag(x_art_complex)

    rolled_stems_B = batch.stem_types_B[-1:] + batch.stem_types_B[:-1]
    rolled_prov_B = batch.provenance_B[-1:] + batch.provenance_B[:-1]
    active_mask = _merge_active_masks(batch.stem_types_A, rolled_stems_B, stems=stems)
    provenance = _merge_provenance(batch.provenance_A, rolled_prov_B)

    return BranchContext(
        name="art",
        x_complex=x_art_complex,
        x_input=x_art_input,
        x_linear_mag=x_art_linear_mag,
        gain=gain_art,
        active_mask=active_mask,
        crop_meta={
            "i_A": meta_A["i"],
            "j_A": meta_A["j"],
            "i_B": meta_B["i"],
            "j_B": meta_B["j"],
            "rolled_from": torch.roll(torch.arange(x_B_complex.size(0), device=x_B_complex.device), shifts=1),
        },
        provenance=provenance,
        effect_params=batch.effect_params_A,
    )


def build_ref_branch(
    batch: SepFPTrainBatch,
    x_AB_complex: torch.Tensor,
    block_size: tuple[int, int],
    mean: float,
    std: float,
    crop_size: int,
    time_stretch: tuple[float, float] | None,
    stems: tuple[str, ...] = STEM_ORDER,
) -> BranchContext:
    if time_stretch is None:
        x_ref_complex, crop_meta = tracked_extract_random_blocks(x_AB_complex, block_size, i=crop_size)
        crop_meta["stretch"] = torch.ones(x_ref_complex.size(0), device=x_AB_complex.device)
        crop_meta["pad_left"] = torch.zeros(x_ref_complex.size(0), dtype=torch.long, device=x_AB_complex.device)
        crop_meta["pad_right"] = torch.zeros(x_ref_complex.size(0), dtype=torch.long, device=x_AB_complex.device)
    else:
        x_ref_complex, crop_meta = tracked_stretch_and_crop(
            x_AB_complex,
            block_size=block_size,
            stretch_factor=time_stretch,
            i=crop_size,
        )

    x_ref_input, gain_ref = normalize_logmag_with_gain(x_ref_complex, mean=mean, std=std, return_gain=True)
    x_ref_linear_mag = complex_to_linear_mag(x_ref_complex)
    active_mask = build_active_mask(batch.stem_types_AB, stems=stems)

    return BranchContext(
        name="ref",
        x_complex=x_ref_complex,
        x_input=x_ref_input,
        x_linear_mag=x_ref_linear_mag,
        gain=gain_ref,
        active_mask=active_mask,
        crop_meta=crop_meta,
        provenance=batch.provenance_AB,
        effect_params=batch.effect_params_AB,
    )
