from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def estimate_latent_stats(
    autoencoder: nn.Module,
    loader: DataLoader,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate per-channel latent mean/std over images and spatial positions."""
    target_device = torch.device(device)
    was_training = autoencoder.training
    autoencoder.eval()

    channel_sum = None
    channel_square_sum = None
    count = 0
    try:
        for batch in loader:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(target_device, non_blocking=True)
            latents = autoencoder.encode(images).to(dtype=torch.float64)
            if latents.ndim != 4:
                raise ValueError("autoencoder.encode must return a 4D NCHW tensor")

            values = latents.permute(1, 0, 2, 3).reshape(latents.shape[1], -1)
            batch_sum = values.sum(dim=1)
            batch_square_sum = values.square().sum(dim=1)
            if channel_sum is None:
                channel_sum = torch.zeros_like(batch_sum)
                channel_square_sum = torch.zeros_like(batch_square_sum)
            elif batch_sum.shape != channel_sum.shape:
                raise ValueError("latent channel count changed between batches")

            channel_sum += batch_sum
            channel_square_sum += batch_square_sum
            count += values.shape[1]
    finally:
        autoencoder.train(was_training)

    if count == 0 or channel_sum is None or channel_square_sum is None:
        raise ValueError("cannot estimate latent statistics from an empty dataset")

    mean = channel_sum / count
    variance = (channel_square_sum / count - mean.square()).clamp_min(0.0)
    std = variance.sqrt().clamp_min(1e-4)
    return mean.float().view(1, -1, 1, 1), std.float().view(1, -1, 1, 1)
