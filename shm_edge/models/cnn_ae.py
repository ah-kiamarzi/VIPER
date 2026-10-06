"""1D convolutional autoencoder (CNN-AE).

The encoder is a stack of four ``ConvBlock``s, each halving the sequence
length. The decoder mirrors it with four ``TransConvBlock``s and a final
``Conv1d`` that maps back to the input channels. The residual block design
follows "High Fidelity Neural Audio Compression" (Defossez et al., 2023).
"""

import torch
from torch import nn


def conv_bn_act(c_in: int, c_out: int, kernel_size: int, padding_mode: str) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel_size, stride=1, padding=kernel_size // 2, padding_mode=padding_mode),
        nn.BatchNorm1d(c_out),
        nn.ELU(),
    )


def tconv_bn_act(channels: int, kernel_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose1d(channels, channels, kernel_size, stride=1, padding=kernel_size // 2),
        nn.BatchNorm1d(channels),
        nn.ELU(),
    )


class ConvBlock(nn.Module):
    """``layers`` x (Conv1d-BN-ELU), plus a 1x1 projection skip, then a strided Conv1d (/2)."""

    def __init__(self, c_in: int, c_out: int, layers: int = 2, kernel_size: int = 3,
                 down_kernel_size: int = 7, padding_mode: str = "replicate"):
        super().__init__()
        self.backbone = nn.Sequential(*[
            conv_bn_act(c_in if i == 0 else c_out, c_out, kernel_size, padding_mode)
            for i in range(layers)
        ])
        self.downsampling = nn.Conv1d(c_out, c_out, down_kernel_size, stride=2,
                                      padding=down_kernel_size // 2, padding_mode=padding_mode)
        self.projection = nn.Conv1d(c_in, c_out, kernel_size=1, padding_mode=padding_mode)

    def forward(self, x):
        x = self.backbone(x) + self.projection(x)
        return self.downsampling(x)


class TransConvBlock(nn.Module):
    """``layers`` x (ConvTranspose1d-BN-ELU), plus an identity skip, then a strided ConvTranspose1d (x2)."""

    def __init__(self, c_in: int, c_out: int, layers: int = 2, kernel_size: int = 3, up_kernel_size: int = 4):
        super().__init__()
        self.backbone = nn.Sequential(*[tconv_bn_act(c_in, kernel_size) for _ in range(layers)])
        self.upsampling = nn.ConvTranspose1d(c_in, c_out, up_kernel_size, stride=2, padding=1)

    def forward(self, x):
        x = self.backbone(x) + x
        return self.upsampling(x)


class Encoder(nn.Module):
    def __init__(self, channels, latent_channels: int, padding_mode: str):
        super().__init__()
        widths = list(channels) + [latent_channels]
        self.conv = nn.Sequential(*[
            ConvBlock(c_in, c_out, padding_mode=padding_mode)
            for c_in, c_out in zip(widths[:-1], widths[1:])
        ])

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, latent_channels: int, out_channels: int, n_blocks: int):
        super().__init__()
        blocks = [TransConvBlock(latent_channels, latent_channels) for _ in range(2)]
        width = latent_channels
        for _ in range(n_blocks - 2):
            blocks.append(TransConvBlock(width, width // 2))
            width //= 2
        blocks.append(nn.Conv1d(width, out_channels, kernel_size=7, padding=3))
        self.deconv = nn.Sequential(*blocks)

    def forward(self, z):
        return self.deconv(z)


class CNNAutoEncoder(nn.Module):
    """CNN-AE. Input and output shape: ``(batch, length)``.

    ``length`` must be a multiple of 16 (four stride-2 stages).

    Args:
        in_channels: number of input channels.
        latent_channels: channels of the latent representation.
        padding_mode: padding of the encoder convolutions. The models in the
            paper were trained with ``"replicate"``.
    """

    def __init__(self, in_channels: int = 1, latent_channels: int = 32, padding_mode: str = "replicate"):
        super().__init__()
        enc_channels = [in_channels, in_channels, 2 * in_channels, 4 * in_channels]
        self.encoder = Encoder(enc_channels, latent_channels, padding_mode)
        self.decoder = Decoder(latent_channels, in_channels, n_blocks=len(enc_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x.unsqueeze(1))).squeeze(1)
