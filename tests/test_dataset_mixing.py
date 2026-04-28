from pathlib import Path

import torch

from sepfp.data.batch_types import StemSource
from sepfp.data.dataset import SepFPDataset


def test_mix_from_stems_preserves_channels_before_effects():
    dataset = object.__new__(SepFPDataset)
    dataset.num_frames = 4

    stems = {
        "vocals": [
            StemSource(
                audio=torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]),
                provenance_id="vocals",
            )
        ],
        "drums": [
            StemSource(
                audio=torch.tensor([[0.5, 0.5, 0.5, 0.5], [1.5, 1.5, 1.5, 1.5]]),
                provenance_id="drums",
            )
        ],
    }

    mix = dataset._mix_from_stems(stems)

    assert mix.shape == (2, 4)
    assert torch.allclose(
        mix,
        torch.tensor([[1.5, 2.5, 3.5, 4.5], [11.5, 21.5, 31.5, 41.5]]),
    )


def test_mix_from_stems_uses_fallback_shape_for_empty_partition():
    dataset = object.__new__(SepFPDataset)
    dataset.num_frames = 4
    fallback = torch.randn(2, 4)

    mix = dataset._mix_from_stems({}, fallback_like=fallback)

    assert mix.shape == fallback.shape
    assert torch.allclose(mix, torch.zeros_like(fallback))


def test_infer_stem_type_uses_parent_directory_for_moises_layout():
    dataset = object.__new__(SepFPDataset)
    dataset.stems = ("vocals", "drums", "bass", "guitar", "piano", "others")
    dataset.stem_aliases = dataset._build_stem_aliases(stems=dataset.stems, stem_aliases=None)

    assert dataset._infer_stem_type(
        Path("/tmp/song/guitar/20418a04-7a65-4cab-b462-5e8ab42f2d01.wav")
    ) == "guitar"
    assert dataset._infer_stem_type(
        Path("/tmp/song/drums/858cb357-7729-4510-8edb-aefc3925cf6d.wav")
    ) == "drums"


def test_infer_stem_type_maps_moises_aliases():
    dataset = object.__new__(SepFPDataset)
    dataset.stems = ("vocals", "drums", "bass", "guitar", "piano", "others")
    dataset.stem_aliases = dataset._build_stem_aliases(stems=dataset.stems, stem_aliases=None)

    assert dataset._infer_stem_type(Path("/tmp/song/percussion/file.wav")) == "drums"
    assert dataset._infer_stem_type(Path("/tmp/song/other_keys/file.wav")) == "others"
