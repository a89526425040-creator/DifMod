import pytest
import torch

from difmod.config import ModelConfig
from difmod.inference import generate_images, load_trained_model
from difmod.model import DifMod


def tiny_model():
    config = ModelConfig(
        image_size=32, text_width=32, model_width=32, transformer_depth=1,
        attention_heads=4, autoencoder_base_channels=8,
    )
    model = DifMod(config)
    model.set_latent_stats(
        torch.tensor([0.1, -0.2, 0.3, 0.0]).view(1, 4, 1, 1),
        torch.tensor([0.8, 1.2, 0.7, 1.5]).view(1, 4, 1, 1),
    )
    return model


def test_checkpoint_round_trip_and_image_generation(tmp_path):
    original = tiny_model()
    checkpoint_path = tmp_path / "difmod.pt"
    torch.save({
        "autoencoder": original.autoencoder.state_dict(),
        "text_encoder": original.text_encoder.state_dict(),
        "flow": original.flow.state_dict(),
        "latent_mean": original.latent_mean.cpu(),
        "latent_std": original.latent_std.cpu(),
        "config": vars(original.config),
    }, checkpoint_path)

    loaded = load_trained_model(checkpoint_path)
    assert not loaded.training
    assert torch.allclose(loaded.latent_mean, original.latent_mean)
    assert torch.allclose(loaded.latent_std, original.latent_std)

    images = generate_images(loaded, ["a small red car"], steps=1, seed=7)
    assert images.shape == (1, 3, 32, 32)
    assert torch.isfinite(images).all()
    assert images.min() >= -1.0 and images.max() <= 1.0


def test_loader_rejects_incomplete_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "broken.pt"
    torch.save({"config": vars(tiny_model().config)}, checkpoint_path)
    with pytest.raises(ValueError, match="incomplete DifMod checkpoint"):
        load_trained_model(checkpoint_path)
