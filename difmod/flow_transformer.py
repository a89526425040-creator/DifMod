from __future__ import annotations

import math

import torch
from torch import nn


def timestep_embedding(timesteps: torch.Tensor, width: int) -> torch.Tensor:
    half = width // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timesteps.float().reshape(-1, 1) * frequencies.reshape(1, -1)
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if width % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


def spatial_embedding(height: int, width: int, channels: int,
                      device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build a resolution-independent 2D sinusoidal embedding for patch positions."""
    if channels % 4:
        raise ValueError("model width must be divisible by 4 for 2D position embeddings")
    axis_channels = channels // 2
    quarter = channels // 4
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(quarter, device=device, dtype=torch.float32)
        / max(quarter - 1, 1)
    )
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    x_angles = x.reshape(-1, 1) * frequencies.reshape(1, -1)
    y_angles = y.reshape(-1, 1) * frequencies.reshape(1, -1)
    position = torch.cat(
        [torch.sin(x_angles), torch.cos(x_angles),
         torch.sin(y_angles), torch.cos(y_angles)],
        dim=-1,
    )
    return position.to(dtype=dtype).unsqueeze(0)


class FlowBlock(nn.Module):
    def __init__(self, width: int, text_width: int, heads: int) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm_cross = nn.LayerNorm(width)
        self.norm_text = nn.LayerNorm(text_width)
        self.cross_attention = nn.MultiheadAttention(
            width, heads, kdim=text_width, vdim=text_width, batch_first=True
        )
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 3),
            nn.GELU(),
            nn.Linear(width * 3, width),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor,
                context_mask: torch.Tensor) -> torch.Tensor:
        q = self.norm_self(x)
        x = x + self.self_attention(q, q, q, need_weights=False)[0]
        normalized_context = self.norm_text(context)
        x = x + self.cross_attention(
            self.norm_cross(x), normalized_context, normalized_context,
            key_padding_mask=~context_mask.bool(), need_weights=False,
        )[0]
        return x + self.mlp(self.norm_mlp(x))


class MobileFlowTransformer(nn.Module):
    """Compact patch-based conditional velocity predictor for latent flow matching."""

    def __init__(self, latent_channels: int = 4, text_width: int = 192,
                 width: int = 192, depth: int = 4, heads: int = 4,
                 patch_size: int = 2) -> None:
        super().__init__()
        if width % heads or text_width % heads:
            raise ValueError("model width and text width must be divisible by heads")
        if width % 4:
            raise ValueError("model width must be divisible by 4 for 2D position embeddings")
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.patch_in = nn.Conv2d(
            latent_channels, width, kernel_size=patch_size, stride=patch_size
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)
        )
        self.blocks = nn.ModuleList(
            [FlowBlock(width, text_width, heads) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(width)
        self.patch_out = nn.Linear(width, latent_channels * patch_size * patch_size)

    def forward(self, noisy_latents: torch.Tensor, timesteps: torch.Tensor,
                text_context: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = noisy_latents.shape
        p = self.patch_size
        if height % p or width % p:
            raise ValueError("latent height and width must be divisible by patch_size")
        x = self.patch_in(noisy_latents).flatten(2).transpose(1, 2)
        grid_height, grid_width = height // p, width // p
        x = x + spatial_embedding(
            grid_height, grid_width, x.shape[-1], x.device, x.dtype
        )
        x = x + self.time_mlp(timestep_embedding(timesteps, x.shape[-1])).unsqueeze(1)
        for block in self.blocks:
            x = block(x, text_context, text_mask)
        patches = self.patch_out(self.final_norm(x))
        patches = patches.reshape(
            batch, grid_height, grid_width, self.latent_channels, p, p
        )
        return patches.permute(0, 3, 1, 4, 2, 5).reshape(
            batch, self.latent_channels, height, width
        )
