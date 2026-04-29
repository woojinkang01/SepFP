import torch

from sepfp.data.batch_types import BranchContext, BranchEffectParams, SepFPTrainBatch, StemSource
from sepfp.data.preprocess import build_ref_branch, tracked_stretch_and_crop
from sepfp.data.targets import build_sep_targets


class IdentityComplexTransform(torch.nn.Module):
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return audio


class FixedComplexTransform(torch.nn.Module):
    def __init__(self, spec: torch.Tensor) -> None:
        super().__init__()
        self.spec = spec

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.spec.to(audio.device).expand(audio.size(0), -1, -1, -1)


def _stem_source(provenance_id: str) -> StemSource:
    return StemSource(audio=torch.ones(16), provenance_id=provenance_id)


def test_tracked_stretch_and_crop_reuses_recorded_padding():
    x = torch.arange(1 * 4 * 6 * 2, dtype=torch.float32).view(1, 4, 6, 2)
    stretch = torch.tensor([2.0])

    first, meta = tracked_stretch_and_crop(
        x,
        block_size=(4, 5),
        stretch_factor=(1.0, 1.0),
        i=0,
        j=0,
        s=stretch,
    )
    replay, replay_meta = tracked_stretch_and_crop(
        x,
        block_size=(4, 5),
        stretch_factor=(1.0, 1.0),
        i=0,
        j=0,
        s=stretch,
        pad_left=meta["pad_left"],
    )
    forced_other_padding, _ = tracked_stretch_and_crop(
        x,
        block_size=(4, 5),
        stretch_factor=(1.0, 1.0),
        i=0,
        j=0,
        s=stretch,
        pad_left=1 - int(meta["pad_left"][0].item()),
    )

    assert torch.equal(first, replay)
    assert torch.equal(meta["pad_left"], replay_meta["pad_left"])
    assert torch.equal(meta["pad_right"], replay_meta["pad_right"])
    assert meta["pad_left"].shape == torch.Size([1])
    assert meta["pad_right"].shape == torch.Size([1])
    assert not torch.equal(first, forced_other_padding)


def test_ref_sep_target_reuses_branch_padding_metadata():
    x_ab_complex = torch.arange(1 * 4 * 6 * 2, dtype=torch.float32).view(1, 4, 6, 2)
    source = _stem_source("song0_drums")
    batch = SepFPTrainBatch(
        mix_A=torch.ones(1, 16),
        mix_B=torch.ones(1, 16),
        mix_AB=torch.ones(1, 16),
        stem_types_A=(("drums",),),
        stem_types_B=((),),
        stem_types_AB=(("drums",),),
        individual_stems_A=({"drums": (source,)},),
        individual_stems_B=({},),
        individual_stems_AB=({"drums": (source,)},),
        effect_params_A=(BranchEffectParams(()),),
        effect_params_B=(BranchEffectParams(()),),
        effect_params_AB=(BranchEffectParams(()),),
        song_ids=("song0",),
        frame_offsets=torch.tensor([0]),
        partition_indices_A=((0,),),
        partition_indices_B=((),),
        partition_indices_AB=((0,),),
        provenance_A=({"drums": ("song0_drums",)},),
        provenance_B=({},),
        provenance_AB=({"drums": ("song0_drums",)},),
    )
    transform = FixedComplexTransform(x_ab_complex)
    ref_ctx = build_ref_branch(
        batch=batch,
        x_AB_complex=x_ab_complex,
        block_size=(4, 5),
        mean=0.0,
        std=1.0,
        crop_size=0,
        time_stretch=(2.0, 2.0),
        stems=("drums",),
    )
    inactive_art_ctx = BranchContext(
        name="art",
        x_complex=ref_ctx.x_complex,
        x_input=ref_ctx.x_input,
        x_linear_mag=ref_ctx.x_linear_mag,
        gain=ref_ctx.gain,
        active_mask=torch.zeros_like(ref_ctx.active_mask),
        crop_meta={},
        provenance=ref_ctx.provenance,
        effect_params=ref_ctx.effect_params,
    )

    _, ref_targets = build_sep_targets(
        batch=batch,
        art_ctx=inactive_art_ctx,
        ref_ctx=ref_ctx,
        vqt_transform=transform,
        apply_effects=None,
        sample_rate=16000,
        block_size=(4, 5),
        mean=0.0,
        std=1.0,
        stems=("drums",),
    )

    assert "pad_left" in ref_ctx.crop_meta
    assert "pad_right" in ref_ctx.crop_meta
    assert torch.allclose(ref_targets["drums"].tensor[0], ref_ctx.x_linear_mag[0])
