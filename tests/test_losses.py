import torch

from sepfp.data.batch_types import StemBatch
from sepfp.losses.multi_positive_infonce import MultiPositiveInfoNCELoss
from sepfp.losses.separation import SeparationLoss


def test_separation_loss_reports_per_stem_means():
    loss_fn = SeparationLoss()
    pred = {
        "bass": StemBatch(sample_idx=torch.tensor([0]), tensor=torch.ones(1, 1, 2, 2)),
        "drums": StemBatch(sample_idx=torch.tensor([0, 1]), tensor=torch.zeros(2, 1, 2, 2)),
    }
    target = {
        "bass": StemBatch(sample_idx=torch.tensor([0]), tensor=torch.zeros(1, 1, 2, 2)),
        "drums": StemBatch(sample_idx=torch.tensor([0, 1]), tensor=torch.stack([torch.ones(1, 2, 2), torch.full((1, 2, 2), 3.0)])),
    }

    output = loss_fn(pred, target)

    assert torch.allclose(output.loss, torch.tensor(5.0 / 3.0))
    assert torch.allclose(output.per_stem_loss["bass"], torch.tensor(1.0))
    assert torch.allclose(output.per_stem_loss["drums"], torch.tensor(2.0))
    assert output.per_stem_count == {"bass": 1, "drums": 2}


def test_multi_positive_infonce_skips_invalid_anchor_and_avoids_nan():
    loss_fn = MultiPositiveInfoNCELoss(temperature=0.1, trainable=False)
    art = {"bass": StemBatch(sample_idx=torch.tensor([0]), tensor=torch.tensor([[1.0, 0.0]]))}
    ref = {"bass": StemBatch(sample_idx=torch.tensor([0]), tensor=torch.tensor([[1.0, 0.0]]))}
    pos_mask = {"bass": torch.tensor([[True]])}
    output = loss_fn(art, ref, pos_mask)
    assert output.n_anchor == 0
    assert torch.isfinite(output.loss)
    assert output.loss.item() == 0.0


def test_multi_positive_infonce_averages_over_valid_stems():
    loss_fn = MultiPositiveInfoNCELoss(temperature=0.1, trainable=False)
    art = {
        "bass": StemBatch(sample_idx=torch.tensor([0]), tensor=torch.tensor([[1.0, 0.0]])),
        "drums": StemBatch(sample_idx=torch.tensor([0, 1]), tensor=torch.tensor([[1.0, 0.0], [1.0, 0.0]])),
    }
    ref = {
        "bass": StemBatch(sample_idx=torch.tensor([0, 1]), tensor=torch.tensor([[1.0, 0.0], [0.0, 1.0]])),
        "drums": StemBatch(
            sample_idx=torch.tensor([0, 1, 2]),
            tensor=torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        ),
    }
    pos_mask = {
        "bass": torch.tensor([[True, False]]),
        "drums": torch.tensor([[True, False, False], [True, False, False]]),
    }

    output = loss_fn(art, ref, pos_mask)

    stem_mean = torch.stack(tuple(output.per_stem_loss.values())).mean()
    anchor_weighted = (
        output.per_stem_loss["bass"] * output.per_stem_anchor_count["bass"]
        + output.per_stem_loss["drums"] * output.per_stem_anchor_count["drums"]
    ) / output.n_anchor

    assert output.per_stem_anchor_count == {"bass": 1, "drums": 2}
    assert torch.allclose(output.loss, stem_mean)
    assert not torch.allclose(output.loss, anchor_weighted)
