from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from sepfp.compat import LightningDataModule
from sepfp.data.batch_types import SepFPRawExample, SepFPTrainBatch, StemSource
from sepfp.data.dataset import SepFPDataset


def _collate_stem_dict(
    examples: list[SepFPRawExample],
    attr: str,
) -> tuple[dict[str, tuple[StemSource, ...]], ...]:
    return tuple(getattr(example, attr) for example in examples)


def sepfp_collate_fn(examples: list[SepFPRawExample]) -> SepFPTrainBatch:
    return SepFPTrainBatch(
        mix_A=torch.stack([example.mix_A for example in examples], dim=0),
        mix_B=torch.stack([example.mix_B for example in examples], dim=0),
        mix_AB=torch.stack([example.mix_AB for example in examples], dim=0),
        stem_types_A=tuple(example.stem_types_A for example in examples),
        stem_types_B=tuple(example.stem_types_B for example in examples),
        stem_types_AB=tuple(example.stem_types_AB for example in examples),
        individual_stems_A=_collate_stem_dict(examples, "individual_stems_A"),
        individual_stems_B=_collate_stem_dict(examples, "individual_stems_B"),
        individual_stems_AB=_collate_stem_dict(examples, "individual_stems_AB"),
        effect_params_A=tuple(example.effect_params_A for example in examples),
        effect_params_B=tuple(example.effect_params_B for example in examples),
        effect_params_AB=tuple(example.effect_params_AB for example in examples),
        song_ids=tuple(example.song_id for example in examples),
        frame_offsets=torch.tensor([example.frame_offset for example in examples], dtype=torch.long),
        partition_indices_A=tuple(example.partition_indices_A for example in examples),
        partition_indices_B=tuple(example.partition_indices_B for example in examples),
        partition_indices_AB=tuple(example.partition_indices_AB for example in examples),
        provenance_A=tuple(example.provenance_A for example in examples),
        provenance_B=tuple(example.provenance_B for example in examples),
        provenance_AB=tuple(example.provenance_AB for example in examples),
    )


class SepFPDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: dict[str, Any],
        dataloader: dict[str, Any],
        validation_dataset: dict[str, Any] | None = None,
        norm_stats: tuple[float, float] | list[float] | None = None,
    ) -> None:
        super().__init__()
        self.dataset_cfg = dataset
        self.validation_dataset_cfg = validation_dataset
        self.norm_stats = tuple(norm_stats) if norm_stats is not None else None

        dataloader = dict(dataloader)
        devices = dataloader.pop("devices", 1)
        if not isinstance(devices, int):
            devices = len(devices)
        batch_size = dataloader.pop("batch_size", 1) // max(devices, 1)
        self.dataloader_kwargs = {"batch_size": batch_size, **dataloader}

        self.dataset: SepFPDataset | None = None
        self.validation_set: SepFPDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        self.dataset = SepFPDataset(**self.dataset_cfg)
        self.validation_set = SepFPDataset(**self.validation_dataset_cfg) if self.validation_dataset_cfg else None

    def train_dataloader(self) -> DataLoader:
        assert self.dataset is not None
        return DataLoader(self.dataset, shuffle=True, collate_fn=sepfp_collate_fn, **self.dataloader_kwargs)

    def val_dataloader(self) -> DataLoader | None:
        if self.validation_set is None:
            return None
        return DataLoader(self.validation_set, shuffle=False, collate_fn=sepfp_collate_fn, **self.dataloader_kwargs)
