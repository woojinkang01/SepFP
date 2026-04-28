from __future__ import annotations

import numpy as np
from scipy.signal import get_window

import torch
import torch.nn as nn


def broadcast_dim(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x[:, None, :]
    if x.dim() == 1:
        return x[None, None, :]
    if x.dim() == 3:
        return x
    raise ValueError("Only support audio tensors with shape (batch, time) or (time)")


def nextpow2(value: float) -> int:
    return int(np.ceil(np.log2(value)))


def create_cqt_kernels(
    q: float,
    fs: int,
    fmin: float,
    n_bins: int,
    bins_per_octave: int,
    norm: int,
    window: str,
    gamma: float,
):
    freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.double(bins_per_octave))
    alpha = 2.0 ** (1.0 / bins_per_octave) - 1.0
    lengths = np.ceil(q * fs / (freqs + gamma / alpha))
    max_len = int(max(lengths))
    fft_len = int(2 ** (np.ceil(np.log2(max_len))))
    kernels = np.zeros((int(n_bins), int(fft_len)), dtype=np.complex64)

    for k in range(int(n_bins)):
        freq = freqs[k]
        length = lengths[k]
        start = int(np.ceil(fft_len / 2.0 - length / 2.0)) - (1 if length % 2 == 1 else 0)
        window_dispatch = get_window(window, int(length), fftbins=True)
        signal = window_dispatch * np.exp(np.r_[-length // 2 : length // 2] * 1j * 2 * np.pi * freq / fs) / length
        kernels[k, start : start + int(length)] = signal / np.linalg.norm(signal, norm) if norm else signal

    return kernels, fft_len, torch.tensor(lengths).float()


class VQT(nn.Module):
    def __init__(
        self,
        sr: int = 16000,
        hop_length: int = 320,
        fmin: float = 27.5,
        n_bins: int = 288,
        bins_per_octave: int = 36,
        gamma: float = 7.0,
        filter_scale: float = 1.0,
        norm: int = 1,
        window: str = "hann",
        pad_mode: str = "reflect",
        center: bool = False,
        trainable: bool = False,
        output_format: str = "Complex",
    ) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.trainable = trainable
        self.output_format = output_format
        q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)
        kernels, kernel_width, lengths = create_cqt_kernels(
            q=q,
            fs=sr,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            norm=norm,
            window=window,
            gamma=gamma,
        )

        self.register_buffer("sqrt_lengths", lengths.sqrt_().unsqueeze_(-1))
        kernels = torch.from_numpy(kernels).unsqueeze(1)
        padding = kernel_width // 2 if center else 0
        self.conv = nn.Conv1d(
            1,
            2 * n_bins,
            kernel_size=kernel_width,
            stride=hop_length,
            padding=padding,
            padding_mode=pad_mode,
            bias=False,
        )
        with torch.no_grad():
            self.conv.weight.copy_(torch.cat((kernels.real, -kernels.imag), dim=0))
            self.conv.weight.requires_grad = trainable

    def forward(self, x: torch.Tensor, output_format: str | None = None) -> torch.Tensor:
        output_format = output_format or self.output_format
        x = broadcast_dim(x)
        cqt = self.conv(x).view(x.size(0), 2, self.n_bins, -1)
        cqt *= self.sqrt_lengths

        if output_format == "Magnitude":
            margin = 1e-8 if self.trainable else 0
            return cqt.pow(2).sum(-3).add(margin).sqrt()
        if output_format == "Complex":
            return cqt.permute(0, 2, 3, 1)
        raise ValueError(f"Invalid output_format: {output_format}")
