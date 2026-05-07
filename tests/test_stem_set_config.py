from pathlib import Path
import importlib.util

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from sepfp.models.sepfp_model import SepFPModel


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = str((ROOT / "configs").resolve())


def _load_train_script():
    spec = importlib.util.spec_from_file_location("train_sepfp_script", ROOT / "scripts" / "train_sepfp.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_validate_stem_contract = _load_train_script()._validate_stem_contract


def _compose_train(stem_set: str):
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        return compose(config_name="train", overrides=[f"stem_set={stem_set}"])


def test_stem_set_six_is_default_train_contract():
    cfg = _compose_train("six")

    expected = ["vocals", "drums", "bass", "guitar", "piano", "others"]
    assert list(cfg.model.stems) == expected
    assert list(cfg.data.dataset.stems) == expected
    assert list(cfg.data.validation_dataset.stems) == expected
    _validate_stem_contract(cfg)


def test_stem_set_four_wires_model_and_data_contract():
    cfg = _compose_train("four")

    expected = ["vocals", "drums", "bass", "others"]
    assert list(cfg.model.stems) == expected
    assert list(cfg.data.dataset.stems) == expected
    assert list(cfg.data.validation_dataset.stems) == expected
    _validate_stem_contract(cfg)


def test_moisesdb_4stem_data_config_uses_four_stem_contract():
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="train", overrides=["stem_set=four", "data=moisesdb_4stem_full_effect"])

    expected = ["vocals", "drums", "bass", "others"]
    assert list(cfg.model.stems) == expected
    assert list(cfg.data.dataset.stems) == expected
    assert list(cfg.data.validation_dataset.stems) == expected
    assert "/multistem/4stem/moisesdb/" in cfg.data.dataset.data_path
    assert "/multistem/4stem/moisesdb/" in cfg.data.validation_dataset.meta_path
    _validate_stem_contract(cfg)


def test_musdb18hq_4stem_data_config_trains_on_musdb_and_validates_on_moisesdb():
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="train", overrides=["stem_set=four", "data=musdb18hq_4stem_train_moisesdb_val"])

    expected = ["vocals", "drums", "bass", "others"]
    assert list(cfg.model.stems) == expected
    assert list(cfg.data.dataset.stems) == expected
    assert list(cfg.data.validation_dataset.stems) == expected
    assert "/multistem/4stem/musdb18hq/" in cfg.data.dataset.data_path
    assert "/multistem/4stem/musdb18hq/meta_train150" in cfg.data.dataset.meta_path
    assert "/multistem/4stem/moisesdb/" in cfg.data.validation_dataset.data_path
    assert "/multistem/4stem/moisesdb/meta_val32_seed0" in cfg.data.validation_dataset.meta_path
    _validate_stem_contract(cfg)


def test_train_4stem_scratch_profile_contract():
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="train_4stem_scratch")

    expected = ["vocals", "drums", "bass", "others"]
    assert cfg.task_name == "train_4stem_scratch"
    assert list(cfg.model.stems) == expected
    assert list(cfg.data.dataset.stems) == expected
    assert list(cfg.data.validation_dataset.stems) == expected
    assert cfg.phase.name == "joint"
    assert cfg.phase.compute_separation is True
    assert cfg.checkpoint.mode == "none"
    assert cfg.logger.wandb.group == "4stem-from-scratch"
    assert "/multistem/4stem/musdb18hq/meta_train150" in cfg.data.dataset.meta_path
    assert "/multistem/4stem/moisesdb/meta_val32_seed0" in cfg.data.validation_dataset.meta_path
    _validate_stem_contract(cfg)


def test_four_stem_model_omits_guitar_and_piano_projectors():
    cfg = _compose_train("four")

    model = SepFPModel(stems=tuple(cfg.model.stems))

    assert tuple(model.projectors.keys()) == ("vocals", "drums", "bass", "others")
    assert tuple(model.evidence.stems) == ("vocals", "drums", "bass", "others")
    assert model.evidence.stem_query.weight.shape[0] == 4


def test_stem_contract_rejects_model_data_mismatch():
    cfg = OmegaConf.create(
        {
            "model": {"stems": ["vocals", "drums", "bass", "others"]},
            "data": {
                "dataset": {"stems": ["vocals", "drums", "bass", "guitar", "piano", "others"]},
                "validation_dataset": {"stems": ["vocals", "drums", "bass", "others"]},
            },
        }
    )

    with pytest.raises(ValueError, match="cfg.model.stems must match cfg.data.dataset.stems"):
        _validate_stem_contract(cfg)
