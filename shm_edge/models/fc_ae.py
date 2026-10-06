"""Fully connected autoencoder (FC-AE).

The encoder halves the feature dimension at every layer until it reaches the
latent size (e.g. 512 -> 256 -> 128 -> 64 -> 32); the decoder mirrors it.
Every layer is ``Linear -> Sigmoid -> BatchNorm1d``, all of which are
supported by the GAP9 NE16 accelerator.
"""

from torch import nn


def _halving_stack(in_dim: int, out_dim: int) -> nn.Sequential:
    layers = []
    dim = in_dim
    while dim != out_dim:
        nxt = dim // 2 if dim > out_dim else dim * 2
        layers += [nn.Linear(dim, nxt), nn.Sigmoid(), nn.BatchNorm1d(nxt)]
        dim = nxt
    return nn.Sequential(*layers)


class FcEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder_net = _halving_stack(input_dim, latent_dim)

    def forward(self, x):
        return self.encoder_net(x)


class FcDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.decoder_net = _halving_stack(latent_dim, output_dim)

    def forward(self, z):
        return self.decoder_net(z)


class FCAutoEncoder(nn.Module):
    """FC-AE. Input and output shape: ``(batch, input_dim)``.

    ``input_dim`` must equal ``latent_dim * 2**k`` for some ``k >= 1``.
    """

    def __init__(self, input_dim: int = 512, latent_dim: int = 32):
        super().__init__()
        ratio = input_dim // latent_dim
        if input_dim % latent_dim or ratio < 2 or ratio & (ratio - 1):
            raise ValueError(
                f"input_dim ({input_dim}) must be latent_dim ({latent_dim}) times a power of two"
            )
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = FcEncoder(input_dim, latent_dim)
        self.decoder = FcDecoder(latent_dim, input_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))
