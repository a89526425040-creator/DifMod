import torch

from difmod.config import ModelConfig
from difmod.model import DifMod
from train_flow import save_checkpoint


def test_flow_checkpoint_saves_resume_state_atomically(tmp_path):
    config = ModelConfig(
        image_size=32, text_width=32, model_width=32, transformer_depth=1,
        attention_heads=4, autoencoder_base_channels=8,
    )
    model = DifMod(config)
    trainable = list(model.text_encoder.parameters()) + list(model.flow.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    path = tmp_path / "nested" / "flow.pt"

    save_checkpoint(path, model, optimizer, scaler, epoch=3)

    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["epoch"] == 3
    assert state["config"] == vars(config)
    assert "optimizer" in state and "scaler" in state
    assert state["latent_mean"].shape == (1, 4, 1, 1)
    assert not path.with_suffix(".pt.tmp").exists()
