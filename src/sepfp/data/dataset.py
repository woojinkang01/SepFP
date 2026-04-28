from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from sepfp.data.batch_types import (
    BranchEffectParams,
    SepFPRawExample,
    StemSource,
)
from sepfp.data.effects import RandomizedEffectChain

try:  # pragma: no cover - optional runtime dependency
    from pedalboard.io import AudioFile
except ImportError:  # pragma: no cover - optional runtime dependency
    AudioFile = None


log = logging.getLogger(__name__)


DEFAULT_STEM_ALIASES: dict[str, tuple[str, ...]] = {
    "vocals": ("vocals", "vocal"),
    "drums": ("drums", "drum", "percussion", "perc"),
    "bass": ("bass",),
    "guitar": ("guitar",),
    "piano": ("piano", "keys", "keyboard"),
    "others": (
        "others",
        "other",
        "other_keys",
        "bowed_strings",
        "strings",
        "wind",
        "brass",
        "synth",
        "fx",
    ),
}


def audio_read(audio_path: Path, frame_offset: int, num_frames: int) -> torch.Tensor:
    if AudioFile is None:  # pragma: no cover - runtime guard
        raise ImportError("pedalboard is required to read SepFP training audio")

    with AudioFile(str(audio_path), "r") as handle:
        handle.seek(frame_offset)
        chunk = handle.read(num_frames)

    return torch.as_tensor(chunk, dtype=torch.float32)


def partition_into_three(indices: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    n = len(indices)
    if n == 1:
        return [int(indices[0])], [], []

    shuffled = np.random.permutation(indices).tolist()
    branch_a = [shuffled.pop()]
    branch_b = [shuffled.pop()]
    branch_c: list[int] = []
    for idx in shuffled:
        random.choice([branch_a, branch_b, branch_c]).append(idx)
    return branch_a, branch_b, branch_c


class SepFPDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        meta_path: str | None = None,
        duration: float = 7.2,
        sample_rate: int = 16000,
        board: list[dict[str, object]] | tuple[dict[str, object], ...] = (),
        duplicate_dataset: int = 1,
        threshold: float = 0.5,
        subset: float = 1.0,
        num_trials: int = 10,
        stems: tuple[str, ...] = ("vocals", "drums", "bass", "guitar", "piano", "others"),
        stem_aliases: dict[str, tuple[str, ...] | list[str]] | None = None,
    ) -> None:
        self.data_path = Path(data_path).resolve()
        self.meta_path = Path(meta_path or data_path).resolve()
        self.duration = duration
        self.sample_rate = sample_rate
        self.num_frames = int(duration * sample_rate)
        self.duplicate_dataset = duplicate_dataset
        self.threshold = threshold
        self.num_trials = num_trials
        self.stems = stems
        self.stem_aliases = self._build_stem_aliases(stems=stems, stem_aliases=stem_aliases)
        self.effect_chain = RandomizedEffectChain(board)

        self.songs = sorted(f.relative_to(self.meta_path) for f in self.meta_path.glob("**/*.npy"))
        if subset < 1.0:
            rng = random.Random(0)
            self.songs = rng.sample(self.songs, int(len(self.songs) * subset))

        self._cache: dict[int, dict[str, object]] = {}

    def __len__(self) -> int:
        return self.duplicate_dataset * len(self.songs)

    def __getitem__(self, idx: int) -> SepFPRawExample:
        idx = idx % len(self.songs)
        cached = self._cache.get(idx)
        if cached is None:
            activations_path = self.meta_path / self.songs[idx]
            activations = np.load(activations_path)
            with activations_path.with_suffix(".txt").open("r") as handle:
                filelist = [line for line in handle.read().splitlines() if line]

            cached = {"activations": activations, "filelist": filelist}
            self._cache[idx] = cached

        activations = cached["activations"]
        filelist = [self.data_path / rel for rel in cached["filelist"]]

        int_duration = math.ceil(self.duration)
        for _ in range(self.num_trials):
            start_idx = np.random.randint(activations.shape[1] - int_duration)
            active = np.sum(activations[:, start_idx : start_idx + int_duration], axis=1) > self.threshold * self.duration
            if np.sum(active) >= min(2, activations.shape[0]):
                break
        else:
            log.warning("Falling back to first stem for silent track %s", self.songs[idx])
            active = np.zeros(activations.shape[0], dtype=bool)
            active[0] = True
            start_idx = 0

        frame_offset = start_idx * self.sample_rate
        candidates, = np.nonzero(active)
        indices_A, indices_B, _ = partition_into_three(candidates)
        indices_AB = candidates.tolist()

        stems_A, provenance_A = self._collect_stems(filelist, indices_A, frame_offset)
        stems_B, provenance_B = self._collect_stems(filelist, indices_B, frame_offset)
        stems_AB, provenance_AB = self._collect_stems(filelist, indices_AB, frame_offset)

        mix_A = self._mix_from_stems(stems_A)
        mix_B = self._mix_from_stems(stems_B, fallback_like=mix_A)
        mix_AB = self._mix_from_stems(stems_AB, fallback_like=mix_A)

        params_A = self.effect_chain.sample_parameters()
        params_B = self.effect_chain.sample_parameters()
        params_AB = self.effect_chain.sample_parameters()

        effected_A = self.effect_chain.apply_with_params(mix_A, self.sample_rate, params_A)
        effected_B = self.effect_chain.apply_with_params(mix_B, self.sample_rate, params_B)
        effected_AB = self.effect_chain.apply_with_params(mix_AB, self.sample_rate, params_AB)

        return SepFPRawExample(
            mix_A=effected_A,
            mix_B=effected_B,
            mix_AB=effected_AB,
            stem_types_A=tuple(stems_A.keys()),
            stem_types_B=tuple(stems_B.keys()),
            stem_types_AB=tuple(stems_AB.keys()),
            individual_stems_A={k: tuple(v) for k, v in stems_A.items()},
            individual_stems_B={k: tuple(v) for k, v in stems_B.items()},
            individual_stems_AB={k: tuple(v) for k, v in stems_AB.items()},
            effect_params_A=params_A,
            effect_params_B=params_B,
            effect_params_AB=params_AB,
            song_id=str(self.songs[idx]),
            frame_offset=frame_offset,
            partition_indices_A=tuple(indices_A),
            partition_indices_B=tuple(indices_B),
            partition_indices_AB=tuple(indices_AB),
            provenance_A=provenance_A,
            provenance_B=provenance_B,
            provenance_AB=provenance_AB,
        )

    def _collect_stems(
        self,
        filelist: list[Path],
        indices: list[int],
        frame_offset: int,
    ) -> tuple[dict[str, list[StemSource]], dict[str, tuple[str, ...]]]:
        grouped: dict[str, list[StemSource]] = {}
        provenance: dict[str, list[str]] = {}
        for stem_idx in indices:
            path = filelist[stem_idx]
            stem_type = self._infer_stem_type(path)
            if stem_type is None:
                continue
            audio = audio_read(path, frame_offset, self.num_frames)
            source = StemSource(audio=audio, provenance_id=path.as_posix())
            grouped.setdefault(stem_type, []).append(source)
            provenance.setdefault(stem_type, []).append(source.provenance_id)
        return grouped, {stem: tuple(ids) for stem, ids in provenance.items()}

    def _mix_from_stems(
        self,
        stems: dict[str, list[StemSource]],
        fallback_like: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not stems:
            if fallback_like is None:
                return torch.zeros((1, self.num_frames), dtype=torch.float32)
            return torch.zeros_like(fallback_like)

        audio_sum = None
        for sources in stems.values():
            for source in sources:
                stem_audio = source.audio
                audio_sum = stem_audio if audio_sum is None else audio_sum + stem_audio
        return audio_sum

    def _infer_stem_type(self, path: Path) -> str | None:
        tokens = self._path_tokens(path)
        for stem in self.stems:
            aliases = self.stem_aliases.get(stem, (stem,))
            if any(alias in tokens for alias in aliases):
                return stem
        return None

    @staticmethod
    def _normalize_token(token: str) -> str:
        return token.strip().lower().replace(" ", "_").replace("-", "_")

    def _path_tokens(self, path: Path) -> set[str]:
        tokens = {
            self._normalize_token(part)
            for part in path.parts
            if part not in (path.anchor, ".", "")
        }
        stem_token = self._normalize_token(path.stem)
        if stem_token:
            tokens.add(stem_token)
        return tokens

    def _build_stem_aliases(
        self,
        stems: tuple[str, ...],
        stem_aliases: dict[str, tuple[str, ...] | list[str]] | None,
    ) -> dict[str, tuple[str, ...]]:
        aliases: dict[str, tuple[str, ...]] = {}
        provided = stem_aliases or {}
        for stem in stems:
            raw_aliases: Iterable[str] = provided.get(stem, DEFAULT_STEM_ALIASES.get(stem, (stem,)))
            normalized = tuple(dict.fromkeys(self._normalize_token(alias) for alias in raw_aliases if alias))
            aliases[stem] = normalized or (self._normalize_token(stem),)
        return aliases
