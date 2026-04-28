from __future__ import annotations

from collections.abc import Iterable

import torch


def _trainable_params(parameters: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    return [parameter for parameter in parameters if parameter.requires_grad]


def build_sepfp_param_groups(
    module,
    lr_sep: float = 3e-6,
    lr_asid: float = 1e-5,
    lr_asid_temperature: float = 1e-5,
    weight_decay: float = 0.01,
) -> list[dict]:
    sep_params = _trainable_params(module.model.encoder.parameters())
    sep_params += _trainable_params(module.model.evidence.parameters())
    sep_params += _trainable_params(module.model.decoder.parameters())
    projector_params = _trainable_params(module.model.projectors.parameters())
    temperature_params = _trainable_params([module.asid_loss.log_temperature])

    groups = [
        {
            "name": "sep",
            "params": sep_params,
            "lr": lr_sep,
            "weight_decay": weight_decay,
        },
        {
            "name": "asid_projectors",
            "params": projector_params,
            "lr": lr_asid,
            "weight_decay": weight_decay,
        },
    ]
    if temperature_params:
        groups.append(
            {
                "name": "asid_temperature",
                "params": temperature_params,
                "lr": lr_asid_temperature,
                "weight_decay": 0.0,
            }
        )

    grouped_param_ids = [id(parameter) for group in groups for parameter in group["params"]]
    if len(grouped_param_ids) != len(set(grouped_param_ids)):
        raise ValueError("Optimizer param groups contain duplicate parameters.")

    expected_param_ids = {id(parameter) for parameter in module.parameters() if parameter.requires_grad}
    missing_param_ids = expected_param_ids - set(grouped_param_ids)
    extra_param_ids = set(grouped_param_ids) - expected_param_ids
    if missing_param_ids or extra_param_ids:
        raise ValueError(
            "Optimizer param groups do not match trainable module parameters: "
            f"missing={len(missing_param_ids)}, extra={len(extra_param_ids)}"
        )

    return groups


def build_sepfp_optimizer(
    module,
    lr_sep: float = 3e-6,
    lr_asid: float = 1e-5,
    lr_asid_temperature: float = 1e-5,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        build_sepfp_param_groups(
            module=module,
            lr_sep=lr_sep,
            lr_asid=lr_asid,
            lr_asid_temperature=lr_asid_temperature,
            weight_decay=weight_decay,
        ),
        betas=betas,
        eps=eps,
    )
