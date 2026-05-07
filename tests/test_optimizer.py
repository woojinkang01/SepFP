import torch

from sepfp.models.sepfp_model import SepFPModel
from sepfp.training.module import SepFPLightningModule
from sepfp.training.optim import build_sepfp_param_groups


def _param_ids(parameters):
    return {id(parameter) for parameter in parameters}


def test_sepfp_optimizer_param_groups_are_disjoint_and_complete():
    module = SepFPLightningModule()
    groups = build_sepfp_param_groups(
        module=module,
        lr_sep=3e-6,
        lr_asid=1e-5,
        lr_asid_temperature=2e-5,
        weight_decay=0.01,
    )
    by_name = {group["name"]: group for group in groups}

    assert set(by_name) == {"sep", "asid_projectors", "asid_temperature"}
    assert by_name["sep"]["lr"] == 3e-6
    assert by_name["asid_projectors"]["lr"] == 1e-5
    assert by_name["asid_temperature"]["lr"] == 2e-5
    assert by_name["asid_temperature"]["weight_decay"] == 0.0

    grouped_ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == _param_ids(parameter for parameter in module.parameters() if parameter.requires_grad)

    sep_ids = _param_ids(by_name["sep"]["params"])
    projector_ids = _param_ids(by_name["asid_projectors"]["params"])
    temperature_ids = _param_ids(by_name["asid_temperature"]["params"])

    assert _param_ids(module.model.encoder.parameters()) <= sep_ids
    assert _param_ids(module.model.evidence.parameters()) <= sep_ids
    assert _param_ids(module.model.decoder.parameters()) <= sep_ids
    assert _param_ids(module.model.projectors.parameters()) == projector_ids
    assert temperature_ids == {id(module.asid_loss.log_temperature)}
    assert not (sep_ids & projector_ids)
    assert not (sep_ids & temperature_ids)
    assert not (projector_ids & temperature_ids)


def test_configure_optimizers_uses_named_param_group_lrs():
    module = SepFPLightningModule()
    configured = module.configure_optimizers()
    optimizer = configured["optimizer"]

    assert isinstance(optimizer, torch.optim.AdamW)
    lrs_by_name = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert lrs_by_name == {
        "sep": 3e-6,
        "asid_projectors": 1e-5,
        "asid_temperature": 1e-5,
    }


def test_asid_only_optimizer_groups_exclude_frozen_encoder_and_decoder():
    module = SepFPLightningModule(
        model=SepFPModel(asid_gradient_route="evidence"),
        compute_separation=False,
        train_encoder=False,
        train_evidence=True,
        train_decoder=False,
        train_projectors=True,
    )
    groups = build_sepfp_param_groups(
        module=module,
        lr_sep=3e-6,
        lr_asid=1e-5,
        lr_asid_temperature=2e-5,
        weight_decay=0.01,
    )
    by_name = {group["name"]: group for group in groups}

    assert set(by_name) == {"asid_evidence", "asid_projectors", "asid_temperature"}
    assert _param_ids(module.model.evidence.parameters()) == _param_ids(by_name["asid_evidence"]["params"])
    assert _param_ids(module.model.projectors.parameters()) == _param_ids(by_name["asid_projectors"]["params"])
    assert not any(parameter.requires_grad for parameter in module.model.encoder.parameters())
    assert not any(parameter.requires_grad for parameter in module.model.decoder.parameters())


def test_asid_only_projector_scope_can_freeze_evidence_and_temperature():
    module = SepFPLightningModule(
        model=SepFPModel(asid_gradient_route="projector_only"),
        compute_separation=False,
        train_encoder=False,
        train_evidence=False,
        train_decoder=False,
        train_projectors=True,
        train_asid_temperature=False,
    )
    groups = build_sepfp_param_groups(module=module)
    by_name = {group["name"]: group for group in groups}

    assert set(by_name) == {"asid_projectors"}
    assert _param_ids(module.model.projectors.parameters()) == _param_ids(by_name["asid_projectors"]["params"])
    assert not any(parameter.requires_grad for parameter in module.model.encoder.parameters())
    assert not any(parameter.requires_grad for parameter in module.model.evidence.parameters())
    assert not any(parameter.requires_grad for parameter in module.model.decoder.parameters())
    assert not module.asid_loss.log_temperature.requires_grad


def test_asid_only_full_scope_can_train_encoder_without_decoder():
    module = SepFPLightningModule(
        model=SepFPModel(asid_gradient_route="full"),
        compute_separation=False,
        train_encoder=True,
        train_evidence=True,
        train_decoder=False,
        train_projectors=True,
        train_asid_temperature=True,
    )
    groups = build_sepfp_param_groups(module=module)
    by_name = {group["name"]: group for group in groups}

    assert set(by_name) == {"asid_encoder", "asid_evidence", "asid_projectors", "asid_temperature"}
    assert _param_ids(module.model.encoder.parameters()) == _param_ids(by_name["asid_encoder"]["params"])
    assert _param_ids(module.model.evidence.parameters()) == _param_ids(by_name["asid_evidence"]["params"])
    assert _param_ids(module.model.projectors.parameters()) == _param_ids(by_name["asid_projectors"]["params"])
    assert not any(parameter.requires_grad for parameter in module.model.decoder.parameters())
    assert module.asid_loss.log_temperature.requires_grad


def test_asid_only_phase_rejects_trainable_decoder():
    try:
        SepFPLightningModule(
            model=SepFPModel(asid_gradient_route="evidence"),
            compute_separation=False,
            train_decoder=True,
        )
    except ValueError as exc:
        assert "freeze the decoder" in str(exc)
    else:
        raise AssertionError("Expected ASID-only phase with trainable decoder to fail")
