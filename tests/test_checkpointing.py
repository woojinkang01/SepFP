from pathlib import Path

import torch
from omegaconf import OmegaConf

from sepfp.models.sepfp_model import SepFPModel
from sepfp.training.checkpointing import load_weights_only_checkpoint, resolve_checkpoint_loading
from sepfp.training.module import SepFPLightningModule


def _write_checkpoint(path: Path, module: SepFPLightningModule) -> None:
    torch.save(
        {
            "state_dict": module.state_dict(),
            "epoch": 123,
            "global_step": 456,
            "optimizer_states": [{"unused": True}],
        },
        path,
    )


def _assert_state_dict_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key


def test_resume_weights_only_loads_model_and_asid_loss_without_changing_freeze_scope(tmp_path):
    source = SepFPLightningModule()
    checkpoint_path = tmp_path / "source.ckpt"
    _write_checkpoint(checkpoint_path, source)

    target = SepFPLightningModule(
        model=SepFPModel(asid_gradient_route="evidence"),
        compute_separation=False,
        train_encoder=False,
        train_evidence=True,
        train_decoder=False,
        train_projectors=True,
        train_asid_temperature=True,
    )
    report = load_weights_only_checkpoint(target, checkpoint_path)

    assert report["loaded_model"] is True
    assert report["loaded_asid_loss"] is True
    _assert_state_dict_equal(source.model.state_dict(), target.model.state_dict())
    _assert_state_dict_equal(source.asid_loss.state_dict(), target.asid_loss.state_dict())
    assert not any(parameter.requires_grad for parameter in target.model.encoder.parameters())
    assert any(parameter.requires_grad for parameter in target.model.evidence.parameters())
    assert not any(parameter.requires_grad for parameter in target.model.decoder.parameters())
    assert any(parameter.requires_grad for parameter in target.model.projectors.parameters())
    assert target.asid_loss.log_temperature.requires_grad


def test_resolve_checkpoint_loading_resume_returns_ckpt_path_without_loading(tmp_path):
    source = SepFPLightningModule()
    checkpoint_path = tmp_path / "source.ckpt"
    _write_checkpoint(checkpoint_path, source)
    target = SepFPLightningModule()
    before = {key: value.clone() for key, value in target.model.state_dict().items()}
    cfg = OmegaConf.create({"mode": "resume", "path": str(checkpoint_path), "strict": True})

    ckpt_path, report = resolve_checkpoint_loading(cfg, target)

    assert ckpt_path == str(checkpoint_path)
    assert report == {"mode": "resume", "path": str(checkpoint_path)}
    _assert_state_dict_equal(before, target.model.state_dict())


def test_resolve_checkpoint_loading_resume_weights_only_returns_no_ckpt_path(tmp_path):
    source = SepFPLightningModule()
    checkpoint_path = tmp_path / "source.ckpt"
    _write_checkpoint(checkpoint_path, source)
    target = SepFPLightningModule()
    cfg = OmegaConf.create(
        {
            "mode": "resume_weights_only",
            "path": str(checkpoint_path),
            "strict": True,
            "load_model": True,
            "load_asid_loss": True,
        }
    )

    ckpt_path, report = resolve_checkpoint_loading(cfg, target)

    assert ckpt_path is None
    assert report is not None
    assert report["mode"] == "resume_weights_only"
    assert report["loaded_model"] is True
    assert report["loaded_asid_loss"] is True
    _assert_state_dict_equal(source.model.state_dict(), target.model.state_dict())


def test_checkpoint_path_is_required_for_loading_modes():
    module = SepFPLightningModule()
    cfg = OmegaConf.create({"mode": "resume_weights_only", "path": None})

    try:
        resolve_checkpoint_loading(cfg, module)
    except ValueError as exc:
        assert "checkpoint.path is required" in str(exc)
    else:
        raise AssertionError("Expected missing checkpoint.path to fail")
