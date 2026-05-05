from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from sepfp.training.probe_callback import EvaluationProbeCallback


class _Logger:
    def __init__(self) -> None:
        self.logged = []

    def log_metrics(self, metrics, step=None) -> None:
        self.logged.append((metrics, step))


def _trainer(epoch: int, *, global_zero: bool = True):
    return SimpleNamespace(
        current_epoch=epoch,
        global_step=123,
        is_global_zero=global_zero,
        loggers=[_Logger()],
        default_root_dir=str(Path.cwd()),
        strategy=SimpleNamespace(barrier=lambda *args, **kwargs: None),
        precision_plugin=None,
    )


def _callback(tmp_path: Path) -> EvaluationProbeCallback:
    return EvaluationProbeCallback(
        eval_repo=tmp_path,
        config_path=tmp_path / "config.yaml",
        every_n_epochs=10,
        inference_batch_size=2,
        retry_batch_sizes=[1],
    )


def test_should_run_uses_one_based_epoch(tmp_path):
    callback = _callback(tmp_path)

    assert not callback._should_run(_trainer(8))
    assert callback._should_run(_trainer(9))
    assert not callback._should_run(_trainer(10))


def test_payload_metrics_are_prefixed(tmp_path):
    callback = _callback(tmp_path)
    payload = {
        "metrics": {
            "probe/calibrated_mAP": 0.5,
            "probe/HR@1": 0.25,
            "ignored_text": "x",
        },
        "runtime": {"elapsed_wall_seconds": 3.0},
    }

    metrics = callback._payload_to_log_metrics(payload, batch_size=2, memory={"after_peak_allocated_mb": 12.0})

    assert metrics["probe/full/calibrated_mAP"] == 0.5
    assert metrics["probe/full/HR@1"] == 0.25
    assert metrics["probe/full/elapsed_wall_seconds"] == 3.0
    assert metrics["probe/full/inference_batch_size"] == 2.0
    assert metrics["probe/full/after_peak_allocated_mb"] == 12.0
    assert metrics["probe/full/status_failed"] == 0.0
