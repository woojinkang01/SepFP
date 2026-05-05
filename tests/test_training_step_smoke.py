import torch

from sepfp.data.batch_types import BranchEffectParams, EffectOp, SepFPTrainBatch, StemSource
from sepfp.models.sepfp_model import SepFPModel
from sepfp.training.module import SepFPLightningModule


class DummyComplexTransform(torch.nn.Module):
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        batch = audio.size(0)
        spec = torch.zeros(batch, 288, 256, 2)
        spec[..., 0] = audio[:, :1].view(batch, 1, 1)
        return spec


class RecordingComplexTransform(DummyComplexTransform):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        self.events.append("transform")
        return super().forward(audio)


class RecordingSepFPModel(SepFPModel):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def forward_branch(self, ctx):
        self.events.append(f"forward:{ctx.name}")
        return super().forward_branch(ctx)


def _stem_source(value: float, provenance_id: str) -> StemSource:
    return StemSource(audio=torch.full((16,), value), provenance_id=provenance_id)


def _make_batch() -> SepFPTrainBatch:
    return SepFPTrainBatch(
        mix_A=torch.randn(2, 16),
        mix_B=torch.randn(2, 16),
        mix_AB=torch.randn(2, 16),
        stem_types_A=(("bass",), ("drums",)),
        stem_types_B=(("vocals",), ("bass",)),
        stem_types_AB=(("bass",), ("bass", "drums")),
        individual_stems_A=(
            {"bass": (_stem_source(1.0, "song0_bass"),)},
            {"drums": (_stem_source(2.0, "song1_drums"),)},
        ),
        individual_stems_B=(
            {"vocals": (_stem_source(3.0, "song0_vocals"),)},
            {"bass": (_stem_source(4.0, "song1_bass"),)},
        ),
        individual_stems_AB=(
            {"bass": (_stem_source(1.0, "song0_bass"),)},
            {"bass": (_stem_source(4.0, "song1_bass"),), "drums": (_stem_source(2.0, "song1_drums"),)},
        ),
        effect_params_A=(BranchEffectParams(()), BranchEffectParams(())),
        effect_params_B=(BranchEffectParams(()), BranchEffectParams(())),
        effect_params_AB=(BranchEffectParams(()), BranchEffectParams(())),
        song_ids=("song0", "song1"),
        frame_offsets=torch.tensor([0, 0]),
        partition_indices_A=((0,), (1,)),
        partition_indices_B=((1,), (0,)),
        partition_indices_AB=((0,), (0, 1)),
        provenance_A=({"bass": ("song0_bass",)}, {"drums": ("song1_drums",)}),
        provenance_B=({"vocals": ("song0_vocals",)}, {"bass": ("song1_bass",)}),
        provenance_AB=({"bass": ("song0_bass",)}, {"bass": ("song1_bass",), "drums": ("song1_drums",)}),
    )


def _single_stem_batch(stem: str) -> SepFPTrainBatch:
    source = _stem_source(1.0, f"song0_{stem}")
    stem_dict = {stem: (source,)}
    provenance = {stem: (source.provenance_id,)}
    return SepFPTrainBatch(
        mix_A=torch.randn(1, 16),
        mix_B=torch.randn(1, 16),
        mix_AB=torch.randn(1, 16),
        stem_types_A=((stem,),),
        stem_types_B=((),),
        stem_types_AB=((stem,),),
        individual_stems_A=(stem_dict,),
        individual_stems_B=({},),
        individual_stems_AB=(stem_dict,),
        effect_params_A=(BranchEffectParams(()),),
        effect_params_B=(BranchEffectParams(()),),
        effect_params_AB=(BranchEffectParams(()),),
        song_ids=("song0",),
        frame_offsets=torch.tensor([0]),
        partition_indices_A=((0,),),
        partition_indices_B=((),),
        partition_indices_AB=((0,),),
        provenance_A=(provenance,),
        provenance_B=({},),
        provenance_AB=(provenance,),
    )


def test_training_step_smoke():
    batch = _make_batch()
    module = SepFPLightningModule(transform=DummyComplexTransform(), time_stretch=None, lambda_sep=100.0, lambda_asid_warmup_epochs=0)
    output = module.shared_step(batch, stage="smoke")
    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.sep_loss)
    assert torch.isfinite(output.asid_loss)
    assert torch.allclose(output.loss, 100.0 * output.sep_loss + output.asid_loss)


def test_sep_targets_are_built_before_model_forward():
    events: list[str] = []
    batch = _make_batch()
    module = SepFPLightningModule(
        model=RecordingSepFPModel(events),
        transform=RecordingComplexTransform(events),
        time_stretch=None,
        lambda_asid_warmup_epochs=0,
    )

    output = module.shared_step(batch, stage="smoke")

    assert torch.isfinite(output.loss)
    first_forward = min(idx for idx, event in enumerate(events) if event.startswith("forward:"))
    last_transform = max(idx for idx, event in enumerate(events) if event == "transform")
    assert last_transform < first_forward


def test_validation_stem_metric_keys_are_fixed_for_sparse_stems():
    module = SepFPLightningModule(transform=DummyComplexTransform(), time_stretch=None, lambda_asid_warmup_epochs=0)
    calls_by_batch: list[list[str]] = []

    def record_log(name, *args, **kwargs):
        _ = args, kwargs
        calls_by_batch[-1].append(name)

    module.log = record_log

    for batch in (_single_stem_batch("bass"), _single_stem_batch("vocals")):
        calls_by_batch.append([])
        output = module.shared_step(batch, stage="val")
        assert torch.isfinite(output.loss)

    stem_metric_keys = [
        key
        for key in calls_by_batch[0]
        if key.startswith("val/loss_raw/sep/")
        or key.startswith("val/sep_count/")
        or key.startswith("val/loss_raw/asid/")
        or key.startswith("val/asid_anchor_count/")
    ]
    assert stem_metric_keys
    assert stem_metric_keys == [
        key
        for key in calls_by_batch[1]
        if key.startswith("val/loss_raw/sep/")
        or key.startswith("val/sep_count/")
        or key.startswith("val/loss_raw/asid/")
        or key.startswith("val/asid_anchor_count/")
    ]
