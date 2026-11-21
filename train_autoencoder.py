from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from difmod.config import ModelConfig
from difmod.data import ImageCaptionDataset
from difmod.latent_stats import estimate_latent_stats
from difmod.model import DifMod
from difmod.objectives import autoencoder_reconstruction_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DifMod's image autoencoder from scratch.")
    parser.add_argument("--manifest", required=True, help="JSONL image-caption manifest")
    parser.add_argument("--output", default="checkpoints/autoencoder.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageCaptionDataset(args.manifest, args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    model = DifMod(ModelConfig(image_size=args.image_size)).to(device)
    optimizer = torch.optim.AdamW(model.autoencoder.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for epoch in range(args.epochs):
        model.autoencoder.train()
        total = 0.0
        for images, _captions in loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                loss = autoencoder_reconstruction_loss(model, images)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.autoencoder.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        print(f"epoch={epoch + 1} reconstruction_loss={total / len(loader):.6f}")

    stats_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    latent_mean, latent_std = estimate_latent_stats(model.autoencoder, stats_loader, device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"autoencoder": model.autoencoder.state_dict(),
                "latent_mean": latent_mean.cpu(), "latent_std": latent_std.cpu(),
                "image_size": args.image_size}, output)
    print(f"latent_mean={latent_mean.flatten().tolist()}")
    print(f"latent_std={latent_std.flatten().tolist()}")
    print(f"saved checkpoint: {output}")


if __name__ == "__main__":
    main()
