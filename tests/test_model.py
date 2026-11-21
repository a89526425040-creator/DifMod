import torch

from difmod.autoencoder import LatentAutoencoder
from difmod.config import ModelConfig
from difmod.flow_transformer import MobileFlowTransformer, spatial_embedding
from difmod.model import DifMod
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


def test_flow_transformer_preserves_latent_shape_with_different_widths():
    batch = 2
    tokenizer = ByteTokenizer(max_length=32)
    tokens, mask = tokenizer.batch_encode(["a small house", "a blue car"])
    text = ByteTextEncoder(vocab_size=259, width=48, max_length=32, layers=1, heads=4)
    context = text(tokens, mask)
    model = MobileFlowTransformer(
        latent_channels=4, text_width=48, width=64, depth=2, heads=4, patch_size=2
    )
    latents = torch.randn(batch, 4, 8, 8)
    times = torch.rand(batch)
    result = model(latents, times, context, mask)
    assert result.shape == latents.shape
    assert torch.isfinite(result).all()


def test_spatial_embedding_is_resolution_aware():
    small = spatial_embedding(4, 4, 64, torch.device("cpu"), torch.float32)
    large = spatial_embedding(8, 8, 64, torch.device("cpu"), torch.float32)
    assert small.shape == (1, 16, 64)
    assert large.shape == (1, 64, 64)
    assert not torch.equal(small[:, :16], large[:, :16])


def test_latent_normalization_round_trip():
    config = ModelConfig(
        image_size=32, text_width=32, model_width=32, transformer_depth=1,
        attention_heads=4, autoencoder_base_channels=8,
    )
    model = DifMod(config)
    mean = torch.tensor([0.2, -0.1, 0.4, 0.0]).view(1, 4, 1, 1)
    std = torch.tensor([0.5, 1.5, 0.8, 2.0]).view(1, 4, 1, 1)
    model.set_latent_stats(mean, std)
    original = torch.randn(2, 4, 4, 4)
    restored = model.denormalize_latents(model.normalize_latents(original))
    assert torch.allclose(original, restored, atol=1e-6)


def test_config_rejects_invalid_dimensions():
    config = ModelConfig(image_size=255)
    try:
        config.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("invalid image_size should be rejected")


def test_config_rejects_incompatible_text_width():
    config = ModelConfig(text_width=190, attention_heads=4)
    try:
        config.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("incompatible text_width should be rejected")
