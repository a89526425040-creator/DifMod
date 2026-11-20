import torch

from difmod.config import ModelConfig
from difmod.flow_transformer import MobileFlowTransformer
from difmod.text_encoder import ByteTextEncoder
from difmod.tokenizer import ByteTokenizer


def test_euler_flow_model_step_keeps_latent_shape():
    tokenizer = ByteTokenizer(max_length=24)
    tokens, mask = tokenizer.batch_encode(["a tiny red car"])
    encoder = ByteTextEncoder(vocab_size=259, width=32, max_length=24, layers=1, heads=4)
    context = encoder(tokens, mask)
    flow = MobileFlowTransformer(
        latent_channels=4, text_width=32, width=32, depth=1, heads=4, patch_size=2
    )
    latents = torch.randn(1, 4, 8, 8)
    velocity = flow(latents, torch.tensor([0.5]), context, mask)
    next_latents = latents + velocity / 8
    assert next_latents.shape == (1, 4, 8, 8)
    assert torch.isfinite(next_latents).all()
