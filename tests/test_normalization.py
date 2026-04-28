import torch

from sepfp.data.preprocess import normalize_logmag_with_gain


def test_normalization_reuses_given_gain():
    x = torch.tensor([[[[3.0, 4.0]]]])
    normalized, gain = normalize_logmag_with_gain(x, mean=0.0, std=1.0, return_gain=True)
    reused = normalize_logmag_with_gain(x, mean=0.0, std=1.0, gain=gain)
    assert torch.allclose(normalized, reused)


def test_normalization_changes_if_gain_is_recomputed_from_different_tensor():
    x = torch.tensor([[[[3.0, 4.0]]]])
    y = torch.tensor([[[[6.0, 8.0]]]])
    _, gain = normalize_logmag_with_gain(x, mean=0.0, std=1.0, return_gain=True)
    reused = normalize_logmag_with_gain(y, mean=0.0, std=1.0, gain=gain)
    recomputed, _ = normalize_logmag_with_gain(y, mean=0.0, std=1.0, return_gain=True)
    assert not torch.allclose(reused, recomputed)
