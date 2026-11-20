import torch

from difmod.autoencoder import LatentAutoencoder
from difmod.config import ModelConfig
from difmod.flow_transformer import MobileFlowTransformer
from difmod.text_encoder import ByteTextEncoder
from difmod.tokenizer import ByteTokenizer


def test_byte_tokenizer_handles_multilingual_text_and_padding():
    tokenizer = ByteTokenizer(max_length=32)
    tokens, mask = tokenizer.batch_encode(["red car", "красная машина"])
    assert tokens.shape == mask.shape
    assert tokens.dtype == torch.long
    assert mask[0].sum() < mask[1].sum()
    assert tokens[0, 0].item() == tokenizer.bos_id


def test_autoencoder_eight_x_spatial_compression():
    model = LatentAutoencoder(latent_channels=4, base_channels=8)
    images = torch.randn(2, 3, 64, 64)
    reconstructed, latents = model(images)
    assert latents.shape == (2, 4, 8, 8)
    assert reconstructed.shape == images.shape


def test_flow_transformer_preserves_latent_shape():
    batch = 2
    tokenizer = ByteTokenizer(max_length=32)
    tokens, mask = tokenizer.batch_encode(["a small house", "a blue car"])
    text = ByteTextEncoder(vocab_size=259, width=64, max_length=32, layers=1, heads=4)
    context = text(tokens, mask)
    model = MobileFlowTransformer(
        latent_channels=4, text_width=64, width=64, depth=2, heads=4, patch_size=2
    )
    latents = torch.randn(batch, 4, 8, 8)
    times = torch.rand(batch)
    result = model(latents, times, context, mask)
    assert result.shape == latents.shape
    assert torch.isfinite(result).all()


def test_config_rejects_invalid_dimensions():
    config = ModelConfig(image_size=255)
    try:
        config.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("invalid image_size should be rejected")
