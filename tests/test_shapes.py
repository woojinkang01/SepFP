import torch

from sepfp.data.batch_types import BranchContext, STEM_ORDER
from sepfp.losses.multi_positive_infonce import MultiPositiveInfoNCELoss
from sepfp.models.sepfp_model import SepFPModel


def _ctx(batch: int, active_mask: torch.BoolTensor) -> BranchContext:
    provenance = []
    for sample_idx in range(batch):
        sample = {}
        for stem_idx, stem in enumerate(STEM_ORDER):
            if active_mask[sample_idx, stem_idx]:
                sample[stem] = (f"song{sample_idx}_{stem}",)
        provenance.append(sample)
    return BranchContext(
        name="art",
        x_complex=torch.randn(batch, 252, 256, 2),
        x_input=torch.randn(batch, 1, 252, 256),
        x_linear_mag=torch.rand(batch, 1, 252, 256).clamp_min(1e-6),
        gain=torch.tensor(1.0),
        active_mask=active_mask,
        crop_meta={
            "i_A": torch.zeros(batch, dtype=torch.long),
            "j_A": torch.zeros(batch, dtype=torch.long),
            "i_B": torch.zeros(batch, dtype=torch.long),
            "j_B": torch.zeros(batch, dtype=torch.long),
            "rolled_from": torch.arange(batch),
        },
        provenance=tuple(provenance),
        effect_params=tuple(),
    )


def test_shape_path_and_absent_stem_skip():
    batch = 2
    active_mask = torch.tensor(
        [
            [True, True, False, False, False, False],
            [False, True, True, False, False, False],
        ],
        dtype=torch.bool,
    )
    ctx = _ctx(batch, active_mask)
    model = SepFPModel()

    features = model.encoder(ctx.x_input)
    assert features.shape == (2, 256, 63, 64)

    outputs = model.forward_branch(ctx)
    assert "guitar" not in outputs.stem_preds
    assert "piano" not in outputs.stem_preds
    assert "others" not in outputs.stem_preds

    for stem_batch in outputs.stem_latents.values():
        assert stem_batch.tensor.shape[1:] == (192, 63, 64)
    for stem_batch in outputs.stem_preds.values():
        assert stem_batch.tensor.shape[1:] == (1, 252, 256)
        assert torch.all(stem_batch.tensor >= 0)
        assert stem_batch.extras["domain"] == "linear_mag"
    for stem_batch in outputs.stem_embeds.values():
        assert stem_batch.tensor.shape[-1] == 512
        assert torch.allclose(stem_batch.tensor.norm(dim=-1), torch.ones(stem_batch.tensor.size(0)), atol=1e-5)


def test_active_softmax_masks_sum_to_one():
    batch = 2
    active_mask = torch.tensor(
        [
            [True, True, False, False, False, False],
            [False, True, True, False, False, False],
        ],
        dtype=torch.bool,
    )
    ctx = _ctx(batch, active_mask)
    model = SepFPModel()
    outputs = model.forward_branch(ctx)

    for sample_idx in range(batch):
        masks = []
        for stem, stem_batch in outputs.stem_preds.items():
            row = torch.nonzero(stem_batch.sample_idx == sample_idx, as_tuple=False).flatten()
            if row.numel() > 0:
                masks.append(stem_batch.extras["mask"][int(row[0].item())])
        mask_sum = torch.stack(masks, dim=0).sum(dim=0)
        assert torch.allclose(mask_sum, torch.ones_like(mask_sum), atol=1e-5)


def test_model_respects_configurable_dimensions():
    batch = 2
    active_mask = torch.tensor(
        [
            [True, False, False, False, False, False],
            [False, True, False, False, False, False],
        ],
        dtype=torch.bool,
    )
    ctx = _ctx(batch, active_mask)
    model = SepFPModel(
        encoder_channels=128,
        evidence_channels=96,
        query_dim=128,
        attention_heads=4,
        decoder_hidden_channels=64,
        projector_hidden_channels=96,
        projector_out_dim=64,
    )

    outputs = model.forward_branch(ctx)

    for stem_batch in outputs.stem_latents.values():
        assert stem_batch.tensor.shape[1:] == (96, 63, 64)
    for stem_batch in outputs.stem_preds.values():
        assert stem_batch.tensor.shape[1:] == (1, 252, 256)
    for stem_batch in outputs.stem_embeds.values():
        assert stem_batch.tensor.shape[-1] == 64


def test_asid_loss_updates_projector_only():
    batch = 2
    active_mask = torch.tensor(
        [
            [True, False, False, False, False, False],
            [True, False, False, False, False, False],
        ],
        dtype=torch.bool,
    )
    ctx = _ctx(batch, active_mask)
    model = SepFPModel()
    outputs = model.forward_branch(ctx)
    pos_masks = {"vocals": torch.eye(batch, dtype=torch.bool)}
    loss = MultiPositiveInfoNCELoss(temperature=0.1, trainable=False)(
        outputs.stem_embeds,
        outputs.stem_embeds,
        pos_masks,
    ).loss

    model.zero_grad(set_to_none=True)
    loss.backward()

    encoder_grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
    evidence_grads = [p.grad for p in model.evidence.parameters() if p.requires_grad]
    decoder_grads = [p.grad for p in model.decoder.parameters() if p.requires_grad]
    projector_grads = [p.grad for p in model.projector.parameters() if p.requires_grad]

    assert all(grad is None for grad in encoder_grads)
    assert all(grad is None for grad in evidence_grads)
    assert all(grad is None for grad in decoder_grads)
    assert any(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in projector_grads)
