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
    assert set(model.projectors.keys()) == set(STEM_ORDER)

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
    model = SepFPModel(mask_mode="active_softmax")
    outputs = model.forward_branch(ctx)

    for sample_idx in range(batch):
        masks = []
        for stem, stem_batch in outputs.stem_preds.items():
            row = torch.nonzero(stem_batch.sample_idx == sample_idx, as_tuple=False).flatten()
            if row.numel() > 0:
                masks.append(stem_batch.extras["mask"][int(row[0].item())])
        mask_sum = torch.stack(masks, dim=0).sum(dim=0)
        assert torch.allclose(mask_sum, torch.ones_like(mask_sum), atol=1e-5)


def test_independent_capped_masks_do_not_force_sum_to_one():
    batch = 2
    active_mask = torch.tensor(
        [
            [True, True, False, False, False, False],
            [False, True, True, False, False, False],
        ],
        dtype=torch.bool,
    )
    ctx = _ctx(batch, active_mask)
    model = SepFPModel(mask_mode="independent_capped", max_mask=2.0)
    outputs = model.forward_branch(ctx)

    for stem_batch in outputs.stem_preds.values():
        mask = stem_batch.extras["mask"]
        assert torch.all(mask >= 0)
        assert torch.all(mask <= 2.0)

    for sample_idx in range(batch):
        masks = []
        for stem_batch in outputs.stem_preds.values():
            row = torch.nonzero(stem_batch.sample_idx == sample_idx, as_tuple=False).flatten()
            if row.numel() > 0:
                masks.append(stem_batch.extras["mask"][int(row[0].item())])
        mask_sum = torch.stack(masks, dim=0).sum(dim=0)
        assert not torch.allclose(mask_sum, torch.ones_like(mask_sum), atol=1e-3)


def test_forward_branch_can_skip_separation_decoder():
    batch = 2
    active_mask = torch.tensor(
        [
            [True, False, False, False, False, False],
            [True, False, False, False, False, False],
        ],
        dtype=torch.bool,
    )
    ctx = _ctx(batch, active_mask)
    model = SepFPModel(mask_mode="independent_capped")
    call_count = 0

    def _count_decoder(_module, _inputs, _output):
        nonlocal call_count
        call_count += 1

    hook = model.decoder.register_forward_hook(_count_decoder)
    try:
        outputs = model.forward_branch(ctx, compute_separation=False)
    finally:
        hook.remove()

    assert call_count == 0
    assert outputs.stem_preds == {}
    assert set(outputs.stem_embeds) == {"vocals"}
    assert torch.allclose(outputs.stem_embeds["vocals"].tensor.norm(dim=-1), torch.ones(batch), atol=1e-5)


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


def _has_grad(parameters) -> bool:
    return any(
        grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        for grad in (parameter.grad for parameter in parameters if parameter.requires_grad)
    )


def _has_no_grad(parameters) -> bool:
    return all(parameter.grad is None for parameter in parameters if parameter.requires_grad)


def test_per_stem_projectors_run_only_for_active_stems():
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
    call_count = {stem: 0 for stem in STEM_ORDER}
    hooks = []

    for stem, projector in model.projectors.items():
        def _count_call(_module, _inputs, _output, stem=stem):
            call_count[stem] += 1

        hooks.append(projector.register_forward_hook(_count_call))

    try:
        model.forward_branch(ctx)
    finally:
        for hook in hooks:
            hook.remove()

    assert call_count["vocals"] == 1
    for stem in STEM_ORDER:
        if stem != "vocals":
            assert call_count[stem] == 0


def test_asid_loss_updates_only_active_stem_projector():
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

    assert _has_no_grad(model.encoder.parameters())
    assert _has_no_grad(model.evidence.parameters())
    assert _has_no_grad(model.decoder.parameters())
    assert _has_grad(model.projectors["vocals"].parameters())
    for stem in STEM_ORDER:
        if stem != "vocals":
            assert _has_no_grad(model.projectors[stem].parameters())


def test_asid_loss_updates_multiple_active_stem_projectors_only():
    batch = 2
    active_mask = torch.tensor(
        [
            [True, True, False, False, False, False],
            [True, True, False, False, False, False],
        ],
        dtype=torch.bool,
    )
    ctx = _ctx(batch, active_mask)
    model = SepFPModel()
    outputs = model.forward_branch(ctx)
    pos_masks = {
        "vocals": torch.eye(batch, dtype=torch.bool),
        "drums": torch.eye(batch, dtype=torch.bool),
    }
    loss = MultiPositiveInfoNCELoss(temperature=0.1, trainable=False)(
        outputs.stem_embeds,
        outputs.stem_embeds,
        pos_masks,
    ).loss

    model.zero_grad(set_to_none=True)
    loss.backward()

    assert _has_no_grad(model.encoder.parameters())
    assert _has_no_grad(model.evidence.parameters())
    assert _has_no_grad(model.decoder.parameters())
    assert _has_grad(model.projectors["vocals"].parameters())
    assert _has_grad(model.projectors["drums"].parameters())
    for stem in ("bass", "guitar", "piano", "others"):
        assert _has_no_grad(model.projectors[stem].parameters())


def test_evidence_asid_gradient_route_updates_evidence_but_not_encoder_or_decoder():
    batch = 2
    active_mask = torch.tensor(
        [
            [True, True, False, False, False, False],
            [True, True, False, False, False, False],
        ],
        dtype=torch.bool,
    )
    ctx = _ctx(batch, active_mask)
    model = SepFPModel(asid_gradient_route="evidence")
    outputs = model.forward_branch(ctx)
    pos_masks = {
        "vocals": torch.eye(batch, dtype=torch.bool),
        "drums": torch.eye(batch, dtype=torch.bool),
    }
    loss = MultiPositiveInfoNCELoss(temperature=0.1, trainable=False)(
        outputs.stem_embeds,
        outputs.stem_embeds,
        pos_masks,
    ).loss

    model.zero_grad(set_to_none=True)
    loss.backward()

    assert _has_no_grad(model.encoder.parameters())
    assert _has_grad(model.evidence.parameters())
    assert _has_no_grad(model.decoder.parameters())
    assert _has_grad(model.projectors["vocals"].parameters())
    assert _has_grad(model.projectors["drums"].parameters())
    for stem in ("bass", "guitar", "piano", "others"):
        assert _has_no_grad(model.projectors[stem].parameters())


def test_full_asid_gradient_route_updates_encoder_and_evidence():
    batch = 2
    active_mask = torch.tensor(
        [
            [True, True, False, False, False, False],
            [True, True, False, False, False, False],
        ],
        dtype=torch.bool,
    )
    ctx = _ctx(batch, active_mask)
    model = SepFPModel(asid_gradient_route="full")
    outputs = model.forward_branch(ctx, compute_separation=False)
    pos_masks = {
        "vocals": torch.eye(batch, dtype=torch.bool),
        "drums": torch.eye(batch, dtype=torch.bool),
    }
    loss = MultiPositiveInfoNCELoss(temperature=0.1, trainable=False)(
        outputs.stem_embeds,
        outputs.stem_embeds,
        pos_masks,
    ).loss

    model.zero_grad(set_to_none=True)
    loss.backward()

    assert _has_grad(model.encoder.parameters())
    assert _has_grad(model.evidence.parameters())
    assert _has_no_grad(model.decoder.parameters())
    assert _has_grad(model.projectors["vocals"].parameters())
    assert _has_grad(model.projectors["drums"].parameters())


def test_asid_gradient_route_does_not_change_state_dict_keys():
    baseline = SepFPModel(asid_gradient_route="projector_only")
    evidence_route = SepFPModel(asid_gradient_route="evidence")
    full_route = SepFPModel(asid_gradient_route="full")

    assert baseline.state_dict().keys() == evidence_route.state_dict().keys()
    assert baseline.state_dict().keys() == full_route.state_dict().keys()


def test_evidence_asid_gradient_route_does_not_double_update_evidence_batchnorm_stats():
    batch = 2
    active_mask = torch.tensor(
        [
            [True, True, False, False, False, False],
            [True, True, False, False, False, False],
        ],
        dtype=torch.bool,
    )
    ctx = _ctx(batch, active_mask)
    baseline = SepFPModel(asid_gradient_route="projector_only")
    evidence_route = SepFPModel(asid_gradient_route="evidence")
    evidence_route.load_state_dict(baseline.state_dict())
    baseline.train()
    evidence_route.train()

    baseline.forward_branch(ctx)
    evidence_route.forward_branch(ctx)

    baseline_buffers = dict(baseline.evidence.named_buffers())
    evidence_route_buffers = dict(evidence_route.evidence.named_buffers())
    for name, baseline_buffer in baseline_buffers.items():
        if "running_" in name or "num_batches_tracked" in name:
            assert torch.equal(baseline_buffer, evidence_route_buffers[name])
