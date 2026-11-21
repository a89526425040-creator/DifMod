from __future__ import annotations

from pathlib import Path

import torch

from .config import ModelConfig
from .model import DifMod


def load_trained_model(checkpoint_path: str | Path,
                       device: str | torch.device = "cpu") -> DifMod:
    """Load a complete DifMod flow checkpoint and return an eval-mode model.

    The checkpoint must contain the autoencoder, text encoder, flow transformer,
    calibrated latent statistics, and the model config saved by train_flow.py.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {
        "autoencoder", "text_encoder", "flow",
        "latent_mean", "latent_std", "config",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(
            "incomplete DifMod checkpoint; missing: " + ", ".join(sorted(missing))
        )
    if not isinstance(checkpoint["config"], dict):
        raise ValueError("checkpoint config must be a dictionary")

    config = ModelConfig(**checkpoint["config"])
    model = DifMod(config)
    model.autoencoder.load_state_dict(checkpoint["autoencoder"], strict=True)
    model.text_encoder.load_state_dict(checkpoint["text_encoder"], strict=True)
    model.flow.load_state_dict(checkpoint["flow"], strict=True)
    model.set_latent_stats(checkpoint["latent_mean"], checkpoint["latent_std"])
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def generate_latents(model: DifMod, prompts: list[str], steps: int = 12,
                     seed: int = 0, device: str | torch.device = "cpu") -> torch.Tensor:
    """Return autoencoder-space latents after integrating in normalized latent space.

    Meaningful output requires a trained text encoder, autoencoder, and flow model.
    """
    if not prompts:
        raise ValueError("at least one prompt is required")
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
    return model.denormalize_latents(latents)


@torch.inference_mode()
def generate_images(model: DifMod, prompts: list[str], steps: int = 12,
                    seed: int = 0, device: str | torch.device = "cpu") -> torch.Tensor:
    latents = generate_latents(model, prompts, steps=steps, seed=seed, device=device)
    return model.autoencoder.decode(latents).clamp(-1.0, 1.0)
