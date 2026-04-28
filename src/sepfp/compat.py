from __future__ import annotations

import torch.nn as nn

try:
    from lightning import LightningDataModule, LightningModule
except ImportError:  # pragma: no cover - fallback for minimal environments
    class LightningModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.current_epoch = 0

        def log(self, *args, **kwargs) -> None:
            return None

        def log_dict(self, *args, **kwargs) -> None:
            return None

    class LightningDataModule:
        pass
