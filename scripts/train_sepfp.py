from __future__ import annotations

import hydra
import lightning as L
import rootutils
from omegaconf import DictConfig
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from sepfp.data.datamodule import SepFPDataModule
from sepfp.data.effects import RandomizedEffectChain
from sepfp.losses.multi_positive_infonce import MultiPositiveInfoNCELoss
from sepfp.losses.separation import SeparationLoss
from sepfp.models.sepfp_model import SepFPModel
from sepfp.training.checkpointing import resolve_checkpoint_loading
from sepfp.training.module import SepFPLightningModule


def _art_time_jitter_frames(cfg: DictConfig) -> int | None:
    crop_cfg = cfg.model.get("crop")
    if crop_cfg is None or crop_cfg.get("art_time_jitter_seconds") is None:
        return None
    seconds = float(crop_cfg.art_time_jitter_seconds)
    sample_rate = int(cfg.data.dataset.sample_rate)
    hop_length = int(cfg.model.transform.hop_length)
    return int(round(seconds * sample_rate / hop_length))


def _stems_tuple(value) -> tuple[str, ...]:
    return tuple(str(stem) for stem in value)


def _validate_stem_contract(cfg: DictConfig) -> None:
    model_stems = _stems_tuple(cfg.model.stems)
    if not model_stems:
        raise ValueError("model.stems must contain at least one stem.")
    if len(model_stems) != len(set(model_stems)):
        raise ValueError(f"model.stems contains duplicate entries: {model_stems}")

    dataset_cfg = cfg.data.dataset
    if dataset_cfg.get("stems") is None:
        raise ValueError("cfg.data.dataset.stems must be set and match cfg.model.stems.")
    dataset_stems = _stems_tuple(dataset_cfg.stems)
    if dataset_stems != model_stems:
        raise ValueError(
            "Stem contract mismatch: cfg.model.stems must match cfg.data.dataset.stems. "
            f"model={model_stems}, dataset={dataset_stems}"
        )

    validation_cfg = cfg.data.get("validation_dataset")
    if validation_cfg is not None:
        if validation_cfg.get("stems") is None:
            raise ValueError("cfg.data.validation_dataset.stems must be set and match cfg.model.stems.")
        validation_stems = _stems_tuple(validation_cfg.stems)
        if validation_stems != model_stems:
            raise ValueError(
                "Stem contract mismatch: cfg.model.stems must match cfg.data.validation_dataset.stems. "
                f"model={model_stems}, validation_dataset={validation_stems}"
            )


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)
    _validate_stem_contract(cfg)

    phase_cfg = cfg.get("phase", {})
    phase_asid_route = phase_cfg.get("asid_gradient_route") if phase_cfg else None
    asid_gradient_route = phase_asid_route or cfg.model.get("asid_gradient_route", "projector_only")
    lambda_sep = cfg.loss.sep.lambda_
    if phase_cfg and phase_cfg.get("lambda_sep") is not None:
        lambda_sep = float(phase_cfg.lambda_sep)

    datamodule = hydra.utils.instantiate(cfg.data)
    effect_chain = RandomizedEffectChain(cfg.data.dataset.board)
    model = SepFPModel(
        stems=tuple(cfg.model.stems),
        encoder_channels=cfg.model.encoder.out_channels,
        evidence_channels=cfg.model.evidence.channels,
        query_dim=cfg.model.evidence.query_dim,
        attention_heads=cfg.model.evidence.num_attention_heads,
        decoder_hidden_channels=cfg.model.decoder.hidden_channels,
        projector_hidden_channels=cfg.model.projector.hidden_channels,
        projector_out_dim=cfg.model.projector.out_dim,
        mask_mode=cfg.model.decoder.get("mask_mode", "active_softmax"),
        max_mask=cfg.model.decoder.get("max_mask", 2.0),
        asid_gradient_route=asid_gradient_route,
    )
    crop_cfg = cfg.model.get("crop", {})
    module = SepFPLightningModule(
        model=model,
        transform=hydra.utils.instantiate(cfg.model.transform),
        sep_loss=SeparationLoss(),
        asid_loss=MultiPositiveInfoNCELoss(
            temperature=cfg.loss.asid.temperature,
            trainable=cfg.loss.asid.trainable,
        ),
        optimizer=hydra.utils.instantiate(cfg.optimizer),
        scheduler=hydra.utils.instantiate(cfg.scheduler) if cfg.get("scheduler") else None,
        sample_rate=cfg.data.dataset.sample_rate,
        block_size=tuple(cfg.model.block_size),
        time_stretch=tuple(cfg.model.time_stretch) if cfg.model.get("time_stretch") is not None else None,
        pitch_shift=cfg.model.pitch_shift,
        lognorm_mean=cfg.data.norm_stats[0],
        lognorm_std=cfg.data.norm_stats[1],
        stems=tuple(cfg.model.stems),
        art_time_crop_mode=crop_cfg.get("art_time_crop_mode", "random"),
        art_max_time_jitter_frames=_art_time_jitter_frames(cfg),
        art_share_time_jitter=bool(crop_cfg.get("art_share_time_jitter", False)),
        ref_time_crop_mode=crop_cfg.get("ref_time_crop_mode", "random"),
        ref_padding_mode=crop_cfg.get("ref_padding_mode", "random"),
        lambda_sep=lambda_sep,
        lambda_asid_final=cfg.loss.asid.lambda_final,
        lambda_asid_warmup_epochs=cfg.loss.asid.warmup_epochs,
        apply_effects=effect_chain.apply_with_params,
        phase_name=phase_cfg.get("name", "joint") if phase_cfg else "joint",
        compute_separation=phase_cfg.get("compute_separation", True) if phase_cfg else True,
        train_encoder=phase_cfg.get("train_encoder", True) if phase_cfg else True,
        train_evidence=phase_cfg.get("train_evidence", True) if phase_cfg else True,
        train_decoder=phase_cfg.get("train_decoder", True) if phase_cfg else True,
        train_projectors=phase_cfg.get("train_projectors", True) if phase_cfg else True,
        train_asid_temperature=phase_cfg.get("train_asid_temperature") if phase_cfg else None,
    )

    callbacks: list[Callback] = []
    if cfg.get("callbacks"):
        for _, callback_cfg in cfg.callbacks.items():
            if callback_cfg and "_target_" in callback_cfg:
                callbacks.append(hydra.utils.instantiate(callback_cfg))

    logger: list[Logger] = []
    if cfg.get("logger"):
        for _, logger_cfg in cfg.logger.items():
            if logger_cfg and "_target_" in logger_cfg:
                logger.append(hydra.utils.instantiate(logger_cfg))

    ckpt_path, checkpoint_report = resolve_checkpoint_loading(cfg.checkpoint, module)
    if checkpoint_report is not None:
        print(f"[SepFP checkpoint] {checkpoint_report}")

    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
