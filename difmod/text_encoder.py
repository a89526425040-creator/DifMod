from __future__ import annotations

import torch
from torch import nn


class ByteTextEncoder(nn.Module):
    """Train-from-scratch text encoder operating on UTF-8 byte tokens."""

    def __init__(self, vocab_size: int = 259, width: int = 192, max_length: int = 128,
                 layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, width, padding_idx=0)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_length, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 3,
            dropout=0.0, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(width)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if tokens.shape != attention_mask.shape:
            raise ValueError("tokens and attention_mask must have identical shapes")
        if tokens.shape[1] > self.position_embedding.shape[1]:
            raise ValueError("input text exceeds configured maximum length")
        x = self.token_embedding(tokens) + self.position_embedding[:, : tokens.shape[1]]
        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        return self.norm(x)
