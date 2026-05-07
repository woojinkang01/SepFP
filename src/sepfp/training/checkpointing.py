from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _state_dict_from_checkpoint(path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)!r}")
    return {str(key): value for key, value in state_dict.items() if torch.is_tensor(value)}


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {key.removeprefix(prefix): value for key, value in state_dict.items() if key.startswith(prefix)}


def load_weights_only_checkpoint(
    module,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
    load_model: bool = True,
    load_asid_loss: bool = True,
) -> dict[str, Any]:
    """Load model/loss weights without restoring optimizer, epoch, or step state."""

    state_dict = _state_dict_from_checkpoint(checkpoint_path)
    report: dict[str, Any] = {
        "path": str(Path(checkpoint_path).expanduser()),
        "loaded_model": False,
        "loaded_asid_loss": False,
        "model_missing_keys": (),
        "model_unexpected_keys": (),
        "asid_loss_missing_keys": (),
        "asid_loss_unexpected_keys": (),
    }

    if load_model:
        model_state = _strip_prefix(state_dict, "model.")
        if not model_state:
            raise KeyError("No model.* keys found in checkpoint")
        incompatible = module.model.load_state_dict(model_state, strict=strict)
        report["loaded_model"] = True
        report["model_missing_keys"] = tuple(incompatible.missing_keys)
        report["model_unexpected_keys"] = tuple(incompatible.unexpected_keys)

    if load_asid_loss:
        asid_loss_state = _strip_prefix(state_dict, "asid_loss.")
        if not asid_loss_state:
            raise KeyError("No asid_loss.* keys found in checkpoint")
        incompatible = module.asid_loss.load_state_dict(asid_loss_state, strict=strict)
        report["loaded_asid_loss"] = True
        report["asid_loss_missing_keys"] = tuple(incompatible.missing_keys)
        report["asid_loss_unexpected_keys"] = tuple(incompatible.unexpected_keys)

    return report


def resolve_checkpoint_loading(checkpoint_cfg, module) -> tuple[str | None, dict[str, Any] | None]:
    mode = str(checkpoint_cfg.get("mode", "none"))
    if mode == "none":
        return None, None
    if mode not in {"resume", "resume_weights_only"}:
        raise ValueError(f"Unsupported checkpoint mode: {mode}")

    path = checkpoint_cfg.get("path")
    if path is None or str(path).strip() == "":
        raise ValueError(f"checkpoint.path is required when checkpoint.mode={mode!r}")
    path = str(Path(str(path)).expanduser())

    if mode == "resume":
        return path, {"mode": mode, "path": path}

    report = load_weights_only_checkpoint(
        module,
        path,
        strict=bool(checkpoint_cfg.get("strict", True)),
        load_model=bool(checkpoint_cfg.get("load_model", True)),
        load_asid_loss=bool(checkpoint_cfg.get("load_asid_loss", True)),
    )
    report["mode"] = mode
    return None, report
