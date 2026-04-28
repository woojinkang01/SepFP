import torch

from sepfp.data.batch_types import BranchEffectParams, EffectOp, SepFPTrainBatch, StemSource
from sepfp.training.module import SepFPLightningModule


class DummyComplexTransform(torch.nn.Module):
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        batch = audio.size(0)
        spec = torch.zeros(batch, 288, 256, 2)
        spec[..., 0] = audio[:, :1].view(batch, 1, 1)
        return spec


def _stem_source(value: float, provenance_id: str) -> StemSource:
    return StemSource(audio=torch.full((16,), value), provenance_id=provenance_id)


def test_training_step_smoke():
    batch = SepFPTrainBatch(
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
    module = SepFPLightningModule(transform=DummyComplexTransform(), time_stretch=None, lambda_sep=100.0, lambda_asid_warmup_epochs=0)
    output = module.shared_step(batch, stage="smoke")
    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.sep_loss)
    assert torch.isfinite(output.asid_loss)
    assert torch.allclose(output.loss, 100.0 * output.sep_loss + output.asid_loss)
