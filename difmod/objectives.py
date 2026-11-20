from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import DifMod


def autoencoder_reconstruction_loss(model: DifMod, images: torch.Tensor) -> torch.Tensor:
    """Train the image autoencoder before using its latents for flow matching."""
    reconstruction, _ = model.autoencoder(images)
    return F.l1_loss(reconstruction, images) + 0.1 * F.mse_loss(reconstruction, images)
