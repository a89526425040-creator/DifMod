from __future__ import annotations

import torch

from .model import DifMod


@torch.inference_mode()
def generate_latents(model: DifMod, prompts: list[str], steps: int = 12,
                     seed: int = 0, device: str | torch.device = "cpu") -> torch.Tensor:
    """Euler-integrate the learned flow field from Gaussian noise to image latents.

    Meaningful output requires a trained text encoder, autoencoder, and flow model.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    model = model.to(device).eval()
    context, mask = model.encode_prompts(prompts, device)
    image_size = model.config.image_size
    latent_size = image_size // 8
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        len(prompts), model.config.latent_channels, latent_size, latent_size,
        generator=generator, device=device,
    )
    dt = 1.0 / steps
    for index in range(steps):
        t_value = index / steps
        times = torch.full((len(prompts),), t_value, device=device)
        velocity = model.flow(latents, times, context, mask)
        latents = latents + velocity * dt
    return latents


@torch.inference_mode()
def generate_images(model: DifMod, prompts: list[str], steps: int = 12,
                    seed: int = 0, device: str | torch.device = "cpu") -> torch.Tensor:
    latents = generate_latents(model, prompts, steps=steps, seed=seed, device=device)
    return model.autoencoder.decode(latents).clamp(-1.0, 1.0)
