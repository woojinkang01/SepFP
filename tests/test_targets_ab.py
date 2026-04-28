import torch

from sepfp.data.preprocess import normalize_logmag_with_gain


def test_ab_target_uses_complex_first_sum():
    a = torch.tensor([[[[1.0, 0.0]]]])
    b = torch.tensor([[[[-1.0, 0.0]]]])
    complex_first = normalize_logmag_with_gain(a + b, mean=0.0, std=1.0, gain=torch.tensor(1.0))
    abs_sum_complex = torch.tensor([[[[2.0, 0.0]]]])
    magnitude_sum = normalize_logmag_with_gain(abs_sum_complex, mean=0.0, std=1.0, gain=torch.tensor(1.0))
    assert not torch.allclose(complex_first, magnitude_sum)
