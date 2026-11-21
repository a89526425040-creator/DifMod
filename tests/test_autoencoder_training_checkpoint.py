import torch

from difmod.config import ModelConfig
from difmod.model import DifMod
from train_autoencoder import save_checkpoint


def test_autoencoder_checkpoint_is_resumable_and_not_flow_ready_until_finalized(tmp_path):
    config = ModelConfig(
        image_size=32, text_width=32, model_width=32, transformer_depth=1,
        attention_heads=4, autoencoder_base_channels=8,
    )
    model = DifMod(config)
    optimizer = torch.optim.AdamW(model.autoencoder.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    path = tmp_path / "nested" / "autoencoder.pt"

    save_checkpoint(path, model, optimizer, scaler, epoch=4)
    state = torch.load(path, map_location="cpu", weights_only=True)

    assert state["epoch"] == 4
    assert state["image_size"] == 32
    assert state["config"] == vars(config)
    assert state["training_complete"] is False
    assert "optimizer" in state and "scaler" in state
    assert state["latent_mean"].shape == (1, 4, 1, 1)
    assert not path.with_suffix(".pt.tmp").exists()

    mean = torch.ones(1, 4, 1, 1)
    std = torch.full((1, 4, 1, 1), 2.0)
    save_checkpoint(path, model, optimizer, scaler, epoch=4,
                    latent_mean=mean, latent_std=std, training_complete=True)
    finalized = torch.load(path, map_location="cpu", weights_only=True)
    assert finalized["training_complete"] is True
    assert torch.equal(finalized["latent_mean"], mean)
    assert torch.equal(finalized["latent_std"], std)
