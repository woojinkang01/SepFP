from __future__ import annotations

import gc
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback


class EvaluationProbeCallback(Callback):
    """Run Sample100 checkpoint-probe validation on the live training model."""

    def __init__(
        self,
        eval_repo: str | Path,
        config_path: str | Path,
        *,
        every_n_epochs: int = 10,
        metric_prefix: str = "probe/full",
        inference_batch_size: int = 2,
        retry_batch_sizes: list[int] | tuple[int, ...] = (1,),
        scoring_device: str = "cpu",
        output_dirname: str = "probe_eval",
        empty_cache_before: bool = True,
        empty_cache_after: bool = True,
        fail_on_error: bool = False,
        log_cuda_memory: bool = True,
        use_trainer_precision_context: bool = True,
    ) -> None:
        super().__init__()
        if every_n_epochs <= 0:
            raise ValueError("every_n_epochs must be positive")
        if inference_batch_size <= 0:
            raise ValueError("inference_batch_size must be positive")
        self.eval_repo = Path(eval_repo).expanduser().resolve()
        self.config_path = Path(config_path).expanduser().resolve()
        self.every_n_epochs = int(every_n_epochs)
        self.metric_prefix = metric_prefix.strip("/")
        self.inference_batch_size = int(inference_batch_size)
        self.retry_batch_sizes = tuple(int(value) for value in retry_batch_sizes if int(value) > 0)
        self.scoring_device = str(scoring_device)
        self.output_dirname = str(output_dirname)
        self.empty_cache_before = bool(empty_cache_before)
        self.empty_cache_after = bool(empty_cache_after)
        self.fail_on_error = bool(fail_on_error)
        self.log_cuda_memory = bool(log_cuda_memory)
        self.use_trainer_precision_context = bool(use_trainer_precision_context)

    def _should_run(self, trainer: L.Trainer) -> bool:
        epoch_num = int(trainer.current_epoch) + 1
        return epoch_num % self.every_n_epochs == 0

    def _barrier(self, trainer: L.Trainer) -> None:
        strategy = getattr(trainer, "strategy", None)
        if strategy is not None and hasattr(strategy, "barrier"):
            try:
                strategy.barrier("evaluation_probe")
            except TypeError:
                strategy.barrier()

    def _ensure_eval_import_path(self) -> None:
        src = self.eval_repo / "src"
        if not src.exists():
            raise FileNotFoundError(f"SepFP_evaluation src directory not found: {src}")
        src_str = str(src)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)

    def _output_path(self, trainer: L.Trainer) -> Path:
        root = Path(getattr(trainer, "default_root_dir", None) or Path.cwd())
        root.mkdir(parents=True, exist_ok=True)
        epoch_num = int(trainer.current_epoch) + 1
        return root / self.output_dirname / f"epoch{epoch_num:03d}-step{int(trainer.global_step):08d}.json"

    def _cuda_memory_snapshot(self, device: torch.device, prefix: str) -> dict[str, float]:
        if not self.log_cuda_memory or device.type != "cuda" or not torch.cuda.is_available():
            return {}
        torch.cuda.synchronize(device)
        return {
            f"{prefix}_allocated_mb": torch.cuda.memory_allocated(device) / (1024 * 1024),
            f"{prefix}_reserved_mb": torch.cuda.memory_reserved(device) / (1024 * 1024),
            f"{prefix}_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
            f"{prefix}_peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 * 1024),
        }

    def _precision_context(self, trainer: L.Trainer):
        if not self.use_trainer_precision_context:
            return nullcontext()
        plugin = getattr(trainer, "precision_plugin", None)
        if plugin is not None and hasattr(plugin, "forward_context"):
            return plugin.forward_context()
        return nullcontext()

    def _log_metrics(self, trainer: L.Trainer, metrics: dict[str, float]) -> None:
        if not metrics:
            return
        for logger in trainer.loggers:
            logger.log_metrics(metrics, step=int(trainer.global_step))

    def _run_probe_once(self, trainer: L.Trainer, pl_module: L.LightningModule, batch_size: int) -> dict[str, Any]:
        self._ensure_eval_import_path()
        from sepfp_eval.probe_eval import evaluate_probe_with_models

        device = pl_module.device
        output_path = self._output_path(trainer)
        was_model_training = pl_module.model.training
        was_transform_training = pl_module.transform.training
        pl_module.model.eval()
        pl_module.transform.eval()
        try:
            with self._precision_context(trainer):
                return evaluate_probe_with_models(
                    sepfp_model=pl_module.model,
                    transform=pl_module.transform,
                    device=device,
                    config_path=self.config_path,
                    inference_batch_size=batch_size,
                    scoring_device=self.scoring_device,
                    output_path=output_path,
                    checkpoint_label="live_training_state",
                    metadata={
                        "source": "lightning_callback",
                        "epoch": int(trainer.current_epoch) + 1,
                        "global_step": int(trainer.global_step),
                        "inference_batch_size": int(batch_size),
                    },
                )
        finally:
            pl_module.model.train(was_model_training)
            pl_module.transform.train(was_transform_training)

    @staticmethod
    def _is_oom(exc: BaseException) -> bool:
        text = str(exc).lower()
        return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in text or "cuda oom" in text

    def _payload_to_log_metrics(self, payload: dict[str, Any], batch_size: int, memory: dict[str, float]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for key, value in payload.get("metrics", {}).items():
            if isinstance(value, (int, float)):
                suffix = key.removeprefix("probe/")
                metrics[f"{self.metric_prefix}/{suffix}"] = float(value)
        runtime = payload.get("runtime", {})
        if isinstance(runtime.get("elapsed_wall_seconds"), (int, float)):
            metrics[f"{self.metric_prefix}/elapsed_wall_seconds"] = float(runtime["elapsed_wall_seconds"])
        metrics[f"{self.metric_prefix}/inference_batch_size"] = float(batch_size)
        metrics[f"{self.metric_prefix}/status_failed"] = 0.0
        for key, value in memory.items():
            metrics[f"{self.metric_prefix}/{key}"] = float(value)
        return metrics

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self._should_run(trainer):
            return

        error: BaseException | None = None
        if trainer.is_global_zero:
            device = pl_module.device
            batch_sizes = (self.inference_batch_size,) + self.retry_batch_sizes
            last_error: BaseException | None = None
            try:
                if device.type == "cuda" and torch.cuda.is_available():
                    if self.empty_cache_before:
                        gc.collect()
                        torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                memory_before = self._cuda_memory_snapshot(device, "before")
                for batch_size in batch_sizes:
                    try:
                        payload = self._run_probe_once(trainer, pl_module, batch_size=batch_size)
                        memory_after = self._cuda_memory_snapshot(device, "after")
                        metrics = self._payload_to_log_metrics(payload, batch_size, {**memory_before, **memory_after})
                        self._log_metrics(trainer, metrics)
                        last_error = None
                        break
                    except RuntimeError as exc:
                        last_error = exc
                        if not self._is_oom(exc):
                            raise
                        if device.type == "cuda" and torch.cuda.is_available():
                            gc.collect()
                            torch.cuda.empty_cache()
                if last_error is not None:
                    raise last_error
            except BaseException as exc:
                error = exc
                self._log_metrics(
                    trainer,
                    {
                        f"{self.metric_prefix}/status_failed": 1.0,
                        f"{self.metric_prefix}/inference_batch_size": float(self.inference_batch_size),
                    },
                )
            finally:
                if device.type == "cuda" and torch.cuda.is_available() and self.empty_cache_after:
                    gc.collect()
                    torch.cuda.empty_cache()

        self._barrier(trainer)
        if error is not None and self.fail_on_error:
            raise error
