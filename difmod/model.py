from __future__ import annotations

import torch
from torch import nn

from .autoencoder import LatentAutoencoder
from .config import ModelConfig
from .flow_transformer import MobileFlowTransformer
from .text_encoder import ByteTextEncoder
from .tokenizer import ByteTokenizer


class DifMod(nn.Module):
    """Untrained, from-scratch text-to-image research model."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.config.validate()
        self.tokenizer = ByteTokenizer(self.config.max_text_length)
        self.text_encoder = ByteTextEncoder(
            vocab_size=self.tokenizer.vocab_size,
            width=self.config.text_width,
            max_length=self.config.max_text_length,
            heads=self.config.attention_heads,
        )
        self.autoencoder = LatentAutoencoder(
            latent_channels=self.config.latent_channels,
            base_channels=self.config.autoencoder_base_channels,
        )
        self.flow = MobileFlowTransformer(
            latent_channels=self.config.latent_channels,
            text_width=self.config.text_width,
            width=self.config.model_width,
            depth=self.config.transformer_depth,
            heads=self.config.attention_heads,
            patch_size=self.config.patch_size,
        )

    def encode_prompts(self, prompts: list[str], device: torch.device | str
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, mask = self.tokenizer.batch_encode(prompts)
        tokens, mask = tokens.to(device), mask.to(device)
        return self.text_encoder(tokens, mask), mask

    def flow_matching_loss(self, images: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        """Compute conditional flow-matching loss; autoencoder must be pretrained first."""
        if images.shape[0] != len(prompts):
            raise ValueError("image batch size must match prompt count")
        with torch.no_grad():
            latents = self.autoencoder.encode(images)
        context, mask = self.encode_prompts(prompts, images.device)
        noise = torch.randn_like(latents)
        t = torch.rand(images.shape[0], device=images.device)
        t_view = t.view(-1, 1, 1, 1)
        interpolated = (1.0 - t_view) * noise + t_view * latents
        target_velocity = latents - noise
        predicted_velocity = self.flow(interpolated, t, context, mask)
        return torch.nn.functional.mse_loss(predicted_velocity, target_velocity)
