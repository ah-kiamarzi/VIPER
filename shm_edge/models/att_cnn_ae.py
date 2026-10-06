"""Attention-convolutional autoencoder (Att-CNN-AE), used as the baseline.

Re-implementation of the self-attention autoencoder in Teng et al.,
"Bridge progressive damage detection using unsupervised learning and
self-attention mechanism", Engineering Structures 301 (2024).

Note: as in the reference implementation, the input is permuted to
``(length, batch, channels)`` before attention, so attention mixes samples
*within a batch*. Evaluate with the same batch size used for training
(``shm_edge.evaluate.reconstruction_errors`` does this through
``batch_size``).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class AttCNNAutoEncoder(nn.Module):
    """Att-CNN-AE. Input and output shape: ``(batch, length)``."""

    def __init__(self, in_channels: int = 1, attention_dim: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.project = nn.Conv1d(in_channels, attention_dim, kernel_size=1)
        self.encoder = nn.Sequential(
            nn.Conv1d(attention_dim, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 64, kernel_size=5, stride=1, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.fc_layer = nn.Linear(32, in_channels)

    @staticmethod
    def attention(q, k, v):
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        return F.softmax(scores, dim=-1) @ v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length = x.shape
        x = self.project(x.unsqueeze(1))       # (B, A, L)
        x = x.permute(2, 0, 1)                 # (L, B, A)
        x = self.attention(x, x, x)
        x = x.permute(1, 2, 0)                 # (B, A, L)
        x = self.decoder(self.encoder(x))      # (B, 32, L)
        x = self.fc_layer(x.reshape(batch * length, 32))
        return x.view(batch, length)
