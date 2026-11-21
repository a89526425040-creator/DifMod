from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from difmod.config import ModelConfig
from difmod.data import ImageCaptionDataset
from difmod.model import DifMod


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DifMod conditional flow model from scratch.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--autoencoder", required=True, help="trained autoencoder checkpoint with latent statistics")
    parser.add_argument("--output", default="checkpoints/difmod_flow.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageCaptionDataset(args.manifest, args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    model = DifMod(ModelConfig(image_size=args.image_size)).to(device)
    autoencoder_state = torch.load(args.autoencoder, map_location="cpu", weights_only=True)
    if "latent_mean" not in autoencoder_state or "latent_std" not in autoencoder_state:
        raise ValueError("autoencoder checkpoint is missing latent statistics; retrain the autoencoder")
    model.autoencoder.load_state_dict(autoencoder_state["autoencoder"])
    model.set_latent_stats(autoencoder_state["latent_mean"], autoencoder_state["latent_std"])
    model.autoencoder.eval()
    for parameter in model.autoencoder.parameters():
        parameter.requires_grad_(False)

    trainable = list(model.text_encoder.parameters()) + list(model.flow.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for epoch in range(args.epochs):
        model.train()
        model.autoencoder.eval()
        total = 0.0
        for images, captions in loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                loss = model.flow_matching_loss(images, captions)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        print(f"epoch={epoch + 1} flow_loss={total / len(loader):.6f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"text_encoder": model.text_encoder.state_dict(),
                "flow": model.flow.state_dict(),
                "autoencoder": model.autoencoder.state_dict(),
                "latent_mean": model.latent_mean.cpu(),
                "latent_std": model.latent_std.cpu(),
                "config": vars(model.config)}, output)
    print(f"saved checkpoint: {output}")


if __name__ == "__main__":
    main()
