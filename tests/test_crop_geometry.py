import torch

from sepfp.data.batch_types import BranchEffectParams, SepFPTrainBatch, StemSource
from sepfp.data.preprocess import build_art_branch, generate_time_crop_indices, tracked_stretch_and_crop


def _stem_source(provenance_id: str) -> StemSource:
    return StemSource(audio=torch.ones(16), provenance_id=provenance_id)


def _batch(batch_size: int = 3) -> SepFPTrainBatch:
    sources = tuple(_stem_source(f"song{i}_drums") for i in range(batch_size))
    stem_maps = tuple({"drums": (source,)} for source in sources)
    provenance = tuple({"drums": (source.provenance_id,)} for source in sources)
    stem_types = tuple(("drums",) for _ in range(batch_size))
    effects = tuple(BranchEffectParams(()) for _ in range(batch_size))
    partitions = tuple((0,) for _ in range(batch_size))
    return SepFPTrainBatch(
        mix_A=torch.ones(batch_size, 16),
        mix_B=torch.ones(batch_size, 16),
        mix_AB=torch.ones(batch_size, 16),
        stem_types_A=stem_types,
        stem_types_B=stem_types,
        stem_types_AB=stem_types,
        individual_stems_A=stem_maps,
        individual_stems_B=stem_maps,
        individual_stems_AB=stem_maps,
        effect_params_A=effects,
        effect_params_B=effects,
        effect_params_AB=effects,
        song_ids=tuple(f"song{i}" for i in range(batch_size)),
        frame_offsets=torch.zeros(batch_size, dtype=torch.long),
        partition_indices_A=partitions,
        partition_indices_B=partitions,
        partition_indices_AB=partitions,
        provenance_A=provenance,
        provenance_B=provenance,
        provenance_AB=provenance,
    )


def test_generate_time_crop_indices_center_and_jitter_bounds():
    torch.manual_seed(0)
    center = generate_time_crop_indices(width=348, block_w=256, batch_size=4, mode="center")
    assert torch.equal(center, torch.full((4,), 46))

    jitter = generate_time_crop_indices(
        width=348,
        block_w=256,
        batch_size=256,
        mode="center_jitter",
        max_jitter_frames=25,
    )
    assert int(jitter.min()) >= 21
    assert int(jitter.max()) <= 71


def test_tracked_stretch_and_crop_can_center_crop_and_center_pad():
    x = torch.arange(1 * 4 * 12 * 2, dtype=torch.float32).view(1, 4, 12, 2)

    _, crop_meta = tracked_stretch_and_crop(
        x,
        block_size=(4, 8),
        stretch_factor=(1.0, 1.0),
        i=0,
        s=torch.tensor([1.0]),
        time_crop_mode="center",
        padding_mode="center",
    )
    assert int(crop_meta["j"][0]) == 2
    assert int(crop_meta["pad_left"][0]) == 0
    assert int(crop_meta["pad_right"][0]) == 0

    _, pad_meta = tracked_stretch_and_crop(
        x,
        block_size=(4, 8),
        stretch_factor=(1.0, 1.0),
        i=0,
        s=torch.tensor([2.0]),
        time_crop_mode="center",
        padding_mode="center",
    )
    assert int(pad_meta["j"][0]) == 0
    assert int(pad_meta["pad_left"][0]) == 1
    assert int(pad_meta["pad_right"][0]) == 1


def test_build_art_branch_shared_time_jitter_follows_b_roll():
    torch.manual_seed(3)
    batch = _batch(batch_size=3)
    x_a = torch.randn(3, 4, 10, 2)
    x_b = torch.randn(3, 4, 10, 2)

    ctx = build_art_branch(
        batch=batch,
        x_A_complex=x_a,
        x_B_complex=x_b,
        block_size=(4, 5),
        mean=0.0,
        std=1.0,
        pitch_shift=False,
        crop_size=0,
        stems=("drums",),
        time_crop_mode="center_jitter",
        max_time_jitter_frames=2,
        share_time_jitter=True,
    )

    j_a = ctx.crop_meta["j_A"]
    j_b = ctx.crop_meta["j_B"]
    assert torch.equal(j_a, j_b.roll(1))
    assert int(j_a.min()) >= 0
    assert int(j_a.max()) <= 4
