from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Small, configurable defaults for the first trainable prototype."""

    image_size: int = 256
    latent_channels: int = 4
    text_vocab_size: int = 258
    max_text_length: int = 128
    text_width: int = 192
    model_width: int = 192
    transformer_depth: int = 4
    attention_heads: int = 4
    patch_size: int = 2
    autoencoder_base_channels: int = 32

    def validate(self) -> None:
        if self.image_size < 32 or self.image_size % 8:
            raise ValueError("image_size must be >= 32 and divisible by 8")
        if self.model_width % self.attention_heads:
            raise ValueError("model_width must be divisible by attention_heads")
        if self.patch_size < 1:
            raise ValueError("patch_size must be positive")
        if self.text_vocab_size != 258:
            raise ValueError("The UTF-8 byte tokenizer currently requires vocab size 258")
