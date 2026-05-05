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
from sepfp.training.module import SepFPLightningModule


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

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
        asid_gradient_route=cfg.model.get("asid_gradient_route", "projector_only"),
    )
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
        lambda_sep=cfg.loss.sep.lambda_,
        lambda_asid_final=cfg.loss.asid.lambda_final,
        lambda_asid_warmup_epochs=cfg.loss.asid.warmup_epochs,
        apply_effects=effect_chain.apply_with_params,
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

    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
