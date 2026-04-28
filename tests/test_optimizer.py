import torch

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
