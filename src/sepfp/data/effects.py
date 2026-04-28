from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Mapping, Sequence

import numpy as np
import torch

from sepfp.data.batch_types import BranchEffectParams, EffectOp

try:  # pragma: no cover - optional runtime dependency
    import pedalboard
except ImportError:  # pragma: no cover - optional runtime dependency
    pedalboard = None


def _cast_scalar(value: str) -> int | float:
    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        return int(value)
    return float(value)


def _constant(value: float) -> float:
    return value


def _normal(mean: float, std: float) -> float:
    return (np.random.randn() + mean) * std


def _interval(low: float, high: float) -> float:
    return low + np.random.random() * (high - low)


@dataclass
class EffectTemplate:
    effect_name: str
    probability: float
    defaults: dict[str, int | float]
    samplers: dict[str, callable]


class RandomizedEffectChain:
    def __init__(self, effects: Sequence[Mapping[str, object]] = ()) -> None:
        self.templates: list[EffectTemplate] = []
        self._pedalboard_available = pedalboard is not None

        for effect_spec in effects:
            spec = dict(effect_spec)
            plugin = spec.pop("effect")
            plugin_name = self._resolve_effect_name(plugin)
            repeats = int(spec.pop("repeats", 1))
            probability = float(spec.pop("p"))
            probability = 1 - (1 - probability) ** (1 / repeats) if repeats != 1 else probability

            defaults: dict[str, int | float] = {}
            samplers: dict[str, callable] = {}
            for param, values in spec.items():
                tokens = str(values).split()
                match tokens:
                    case ["choice", default, *choices]:
                        defaults[param] = _cast_scalar(default)
                        choices_array = np.array([_cast_scalar(choice) for choice in choices])
                        samplers[param] = partial(np.random.choice, choices_array)
                    case ["normal", mean, std]:
                        defaults[param] = _cast_scalar(mean)
                        samplers[param] = partial(_normal, _cast_scalar(mean), _cast_scalar(std))
                    case ["uniform", default, low, high]:
                        defaults[param] = _cast_scalar(default)
                        samplers[param] = partial(np.random.randint, _cast_scalar(low), _cast_scalar(high) + 1)
                    case ["random", default, low, high]:
                        defaults[param] = _cast_scalar(default)
                        samplers[param] = partial(_interval, _cast_scalar(low), _cast_scalar(high))
                    case [constant]:
                        defaults[param] = _cast_scalar(constant)
                        samplers[param] = partial(_constant, _cast_scalar(constant))
                    case _:
                        raise ValueError(f"Unsupported pedalboard sampling spec: {values}")

            for _ in range(repeats):
                self.templates.append(
                    EffectTemplate(
                        effect_name=plugin_name,
                        probability=probability,
                        defaults=defaults,
                        samplers=samplers,
                    )
                )

    def sample_parameters(self) -> BranchEffectParams:
        ops: list[EffectOp] = []
        for template in self.templates:
            active = np.random.random() <= template.probability
            params = template.defaults.copy()
            if active:
                params = {name: sampler() for name, sampler in template.samplers.items()}
            ops.append(EffectOp(name=template.effect_name, params=params))
        return BranchEffectParams(tuple(ops))

    def apply_with_params(
        self,
        audio: np.ndarray | torch.Tensor,
        sample_rate: int,
        params: BranchEffectParams,
    ) -> torch.Tensor:
        audio_tensor = torch.as_tensor(audio, dtype=torch.float32)
        if not params.ops or not self._pedalboard_available:
            return self._to_mono(audio_tensor)

        chain = pedalboard.Pedalboard()
        for op in params.ops:
            plugin_cls = getattr(pedalboard, op.name, None)
            if plugin_cls is None:
                raise ValueError(f"Pedalboard plugin {op.name} is unavailable")
            plugin = plugin_cls()
            for key, value in op.params.items():
                setattr(plugin, key, value)
            chain.append(plugin)

        effected = chain(audio_tensor.detach().cpu().numpy(), sample_rate)
        return self._to_mono(torch.as_tensor(effected, dtype=torch.float32))

    @staticmethod
    def _to_mono(audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 1:
            return audio
        return audio.mean(dim=0)

    @staticmethod
    def _resolve_effect_name(plugin: object) -> str:
        if isinstance(plugin, Mapping):
            target = plugin.get("_target_")
            if not target:
                raise ValueError(f"Effect mapping must include _target_, got {plugin}")
            return str(target).split(".")[-1]
        if isinstance(plugin, str):
            return plugin.split(".")[-1]
        if isinstance(plugin, type):
            return plugin.__name__
        return plugin.__class__.__name__
