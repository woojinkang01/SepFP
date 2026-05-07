import pytest
import torch

from sepfp.data.batch_types import BranchEffectParams, BranchContext, EffectOp, SepFPTrainBatch, StemSource
from sepfp.data.preprocess import build_ref_branch, complex_to_linear_mag
from sepfp.data.targets import _apply_effect_to_sources, build_sep_targets


class DummyComplexTransform(torch.nn.Module):
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        batch = audio.size(0)
        spec = torch.zeros(batch, 288, 256, 2)
        spec[..., 0] = audio[:, :1].view(batch, 1, 1)
        return spec


class DeviceCheckingComplexTransform(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("probe", torch.empty(()))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.device != self.probe.device:
            raise RuntimeError(f"device mismatch: audio={audio.device}, transform={self.probe.device}")
        batch = audio.size(0)
        spec = torch.zeros(batch, 288, 256, 2, device=audio.device)
        spec[..., 0] = audio[:, :1].view(batch, 1, 1)
        return spec


def _stem_source(value: float, provenance_id: str) -> StemSource:
    return StemSource(audio=torch.full((16,), value), provenance_id=provenance_id)


def test_complex_to_linear_mag_is_nonnegative_magnitude():
    complex_spec = torch.tensor([[[[3.0, 4.0], [5.0, 12.0]]]])
    mag = complex_to_linear_mag(complex_spec)
    expected = torch.tensor([[[[5.0, 13.0]]]])
    assert torch.equal(mag, expected)


def test_branch_keeps_lognorm_input_and_linear_mag_carrier_separate():
    batch = SepFPTrainBatch(
        mix_A=torch.zeros(1, 16),
        mix_B=torch.zeros(1, 16),
        mix_AB=torch.ones(1, 16) * 2,
        stem_types_A=(("bass",),),
        stem_types_B=(("drums",),),
        stem_types_AB=(("bass",),),
        individual_stems_A=({"bass": (_stem_source(1.0, "song0_bass"),)},),
        individual_stems_B=({"drums": (_stem_source(2.0, "song0_drums"),)},),
        individual_stems_AB=({"bass": (_stem_source(2.0, "song0_bass"),)},),
        effect_params_A=(BranchEffectParams(()),),
        effect_params_B=(BranchEffectParams(()),),
        effect_params_AB=(BranchEffectParams(()),),
        song_ids=("song0",),
        frame_offsets=torch.tensor([0]),
        partition_indices_A=((0,),),
        partition_indices_B=((1,),),
        partition_indices_AB=((0,),),
        provenance_A=({"bass": ("song0_bass",)},),
        provenance_B=({"drums": ("song0_drums",)},),
        provenance_AB=({"bass": ("song0_bass",)},),
    )
    transform = DummyComplexTransform()
    x_ab_complex = transform(batch.mix_AB)
    ctx = build_ref_branch(
        batch=batch,
        x_AB_complex=x_ab_complex,
        block_size=(252, 256),
        mean=0.0,
        std=1.0,
        crop_size=0,
        time_stretch=None,
    )

    assert ctx.x_input.shape == (1, 1, 252, 256)
    assert ctx.x_linear_mag.shape == (1, 1, 252, 256)
    assert torch.all(ctx.x_linear_mag >= 0)
    assert not torch.equal(ctx.x_input, ctx.x_linear_mag)


def test_sep_targets_are_linear_magnitude_not_lognorm():
    batch = SepFPTrainBatch(
        mix_A=torch.zeros(1, 16),
        mix_B=torch.zeros(1, 16),
        mix_AB=torch.ones(1, 16) * 3,
        stem_types_A=(("bass",),),
        stem_types_B=(("drums",),),
        stem_types_AB=(("bass",),),
        individual_stems_A=({"bass": (_stem_source(1.0, "song0_bass"),)},),
        individual_stems_B=({"drums": (_stem_source(2.0, "song0_drums"),)},),
        individual_stems_AB=({"bass": (_stem_source(3.0, "song0_bass"),)},),
        effect_params_A=(BranchEffectParams(()),),
        effect_params_B=(BranchEffectParams(()),),
        effect_params_AB=(BranchEffectParams(()),),
        song_ids=("song0",),
        frame_offsets=torch.tensor([0]),
        partition_indices_A=((0,),),
        partition_indices_B=((1,),),
        partition_indices_AB=((0,),),
        provenance_A=({"bass": ("song0_bass",)},),
        provenance_B=({"drums": ("song0_drums",)},),
        provenance_AB=({"bass": ("song0_bass",)},),
    )
    transform = DummyComplexTransform()
    x_ab_complex = transform(batch.mix_AB)
    ref_ctx = build_ref_branch(
        batch=batch,
        x_AB_complex=x_ab_complex,
        block_size=(252, 256),
        mean=0.0,
        std=1.0,
        crop_size=0,
        time_stretch=None,
    )
    art_ctx = BranchContext(
        name="art",
        x_complex=ref_ctx.x_complex,
        x_input=ref_ctx.x_input,
        x_linear_mag=ref_ctx.x_linear_mag,
        gain=ref_ctx.gain,
        active_mask=ref_ctx.active_mask,
        crop_meta={
            "i_A": torch.zeros(1, dtype=torch.long),
            "j_A": torch.zeros(1, dtype=torch.long),
            "i_B": torch.zeros(1, dtype=torch.long),
            "j_B": torch.zeros(1, dtype=torch.long),
            "rolled_from": torch.arange(1),
        },
        provenance=ref_ctx.provenance,
        effect_params=ref_ctx.effect_params,
    )

    art_targets, ref_targets = build_sep_targets(
        batch=batch,
        art_ctx=art_ctx,
        ref_ctx=ref_ctx,
        vqt_transform=transform,
        apply_effects=None,
        sample_rate=16000,
        block_size=(252, 256),
        mean=-10.0,
        std=99.0,
    )

    for target_batch in (art_targets["bass"], ref_targets["bass"]):
        assert target_batch.tensor.shape == (1, 1, 252, 256)
        assert torch.all(target_batch.tensor >= 0)
        assert torch.isfinite(target_batch.tensor).all()
        assert torch.allclose(target_batch.tensor, torch.full_like(target_batch.tensor, 1.0 if target_batch is art_targets["bass"] else 3.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required to reproduce the DDP target-device path")
def test_sep_targets_move_nested_stem_audio_to_batch_device():
    device = torch.device("cuda")
    batch = SepFPTrainBatch(
        mix_A=torch.zeros(1, 16, device=device),
        mix_B=torch.zeros(1, 16, device=device),
        mix_AB=torch.ones(1, 16, device=device) * 3,
        stem_types_A=(("bass",),),
        stem_types_B=((),),
        stem_types_AB=(("bass",),),
        individual_stems_A=({"bass": (_stem_source(1.0, "song0_bass"),)},),
        individual_stems_B=({},),
        individual_stems_AB=({"bass": (_stem_source(3.0, "song0_bass"),)},),
        effect_params_A=(BranchEffectParams(()),),
        effect_params_B=(BranchEffectParams(()),),
        effect_params_AB=(BranchEffectParams(()),),
        song_ids=("song0",),
        frame_offsets=torch.tensor([0], device=device),
        partition_indices_A=((0,),),
        partition_indices_B=((),),
        partition_indices_AB=((0,),),
        provenance_A=({"bass": ("song0_bass",)},),
        provenance_B=({},),
        provenance_AB=({"bass": ("song0_bass",)},),
    )
    transform = DeviceCheckingComplexTransform().to(device)
    x_ab_complex = transform(batch.mix_AB)
    ref_ctx = build_ref_branch(
        batch=batch,
        x_AB_complex=x_ab_complex,
        block_size=(252, 256),
        mean=0.0,
        std=1.0,
        crop_size=0,
        time_stretch=None,
        stems=("bass",),
    )
    art_ctx = BranchContext(
        name="art",
        x_complex=ref_ctx.x_complex,
        x_input=ref_ctx.x_input,
        x_linear_mag=ref_ctx.x_linear_mag,
        gain=ref_ctx.gain,
        active_mask=ref_ctx.active_mask,
        crop_meta={
            "i_A": torch.zeros(1, dtype=torch.long, device=device),
            "j_A": torch.zeros(1, dtype=torch.long, device=device),
            "i_B": torch.zeros(1, dtype=torch.long, device=device),
            "j_B": torch.zeros(1, dtype=torch.long, device=device),
            "rolled_from": torch.arange(1, device=device),
        },
        provenance=ref_ctx.provenance,
        effect_params=ref_ctx.effect_params,
    )

    art_targets, ref_targets = build_sep_targets(
        batch=batch,
        art_ctx=art_ctx,
        ref_ctx=ref_ctx,
        vqt_transform=transform,
        apply_effects=None,
        sample_rate=16000,
        block_size=(252, 256),
        mean=0.0,
        std=1.0,
        stems=("bass",),
    )

    assert art_targets["bass"].tensor.device.type == device.type
    assert ref_targets["bass"].tensor.device.type == device.type


def test_effect_target_replay_keeps_source_on_cpu_until_after_pedalboard():
    target_device = torch.device("meta")
    seen_devices = []
    seen_dims = []

    def apply_effects(audio: torch.Tensor, sample_rate: int, params: BranchEffectParams) -> torch.Tensor:
        _ = sample_rate, params
        seen_devices.append(audio.device.type)
        seen_dims.append(audio.ndim)
        return audio.mean(dim=0) if audio.ndim > 1 else audio

    stereo_source = StemSource(audio=torch.ones(2, 16), provenance_id="song0_bass")
    effect_params = BranchEffectParams((EffectOp(name="FakeEffect", params={}),))

    outputs = _apply_effect_to_sources(
        sources=(stereo_source,),
        sample_rate=16000,
        effect_params=effect_params,
        apply_effects=apply_effects,
        device=target_device,
    )

    assert seen_devices == ["cpu"]
    assert seen_dims == [2]
    assert len(outputs) == 1
    assert outputs[0].device.type == target_device.type
