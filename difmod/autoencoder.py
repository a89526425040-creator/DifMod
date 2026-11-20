from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2 if downsample else 1, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LatentAutoencoder(nn.Module):
    """Small deterministic 8x image/latent autoencoder.

    This is an autoencoder, not a variational autoencoder. It must be trained
    before its latent space is useful for diffusion.
    """

    def __init__(self, latent_channels: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.encoder = nn.Sequential(
            ConvBlock(3, c1, downsample=True),
            ConvBlock(c1, c2, downsample=True),
            ConvBlock(c2, c3, downsample=True),
            nn.Conv2d(c3, latent_channels, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, c3, 3, padding=1),
            nn.GroupNorm(min(8, c3), c3),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(c3, c2),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(c2, c1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c1, 3, 3, padding=1),
            nn.Tanh(),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(images)
        return self.decode(latents), latents
