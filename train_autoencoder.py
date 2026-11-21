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


def save_checkpoint(path: Path, model: DifMod, optimizer: torch.optim.Optimizer,
                    scaler: torch.amp.GradScaler, epoch: int, *,
                    latent_mean: torch.Tensor | None = None,
                    latent_std: torch.Tensor | None = None,
                    training_complete: bool = False) -> None:
    """Atomically save resumable AE state; only finalized checkpoints are flow-ready."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    mean = model.latent_mean if latent_mean is None else latent_mean
    std = model.latent_std if latent_std is None else latent_std
    torch.save({
        "autoencoder": model.autoencoder.state_dict(),
        "latent_mean": mean.detach().cpu(),
        "latent_std": std.detach().cpu(),
        "image_size": model.config.image_size,
        "config": vars(model.config),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "training_complete": training_complete,
    }, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DifMod's image autoencoder from scratch.")
    parser.add_argument("--manifest", required=True, help="JSONL image-caption manifest")
    parser.add_argument("--output", default="checkpoints/autoencoder.pt")
    parser.add_argument("--resume", action="store_true", help="resume model and optimizer from --output")
    parser.add_argument("--epochs", type=int, default=10, help="total target epoch count, including resumed epochs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.lr <= 0:
        raise ValueError("epochs/batch-size/lr must be positive and workers cannot be negative")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageCaptionDataset(args.manifest, args.image_size)
    if len(dataset) == 0:
        raise ValueError("training dataset is empty")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    model = DifMod(ModelConfig(image_size=args.image_size)).to(device)
    optimizer = torch.optim.AdamW(model.autoencoder.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output = Path(args.output)
    start_epoch = 0

    if args.resume:
        if not output.is_file():
            raise FileNotFoundError(f"cannot resume: checkpoint not found: {output}")
        state = torch.load(output, map_location="cpu", weights_only=True)
        required = {"autoencoder", "optimizer", "scaler", "epoch", "image_size"}
        missing = required.difference(state)
        if missing:
            raise ValueError("resume checkpoint missing: " + ", ".join(sorted(missing)))
        if state["image_size"] != args.image_size:
            raise ValueError("resume checkpoint image_size does not match --image-size")
        if state.get("config", vars(model.config)) != vars(model.config):
            raise ValueError("resume checkpoint model config does not match current arguments")
        model.autoencoder.load_state_dict(state["autoencoder"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"])
        if start_epoch < 0:
            raise ValueError("resume checkpoint has an invalid epoch")
        print(f"resumed from epoch {start_epoch}")
        if start_epoch > args.epochs:
            raise ValueError(
                f"checkpoint is already at epoch {start_epoch}, beyond target --epochs {args.epochs}"
            )

    for epoch in range(start_epoch, args.epochs):
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
        average_loss = total / len(loader)
        save_checkpoint(output, model, optimizer, scaler, epoch + 1)
        print(f"epoch={epoch + 1} reconstruction_loss={average_loss:.6f} checkpoint={output}")

    # A crash after the last epoch save can safely resume and reach this finalization step.
    stats_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    model.autoencoder.eval()
    latent_mean, latent_std = estimate_latent_stats(model.autoencoder, stats_loader, device)
    save_checkpoint(output, model, optimizer, scaler, args.epochs,
                    latent_mean=latent_mean, latent_std=latent_std,
                    training_complete=True)
    print(f"latent_mean={latent_mean.flatten().tolist()}")
    print(f"latent_std={latent_std.flatten().tolist()}")
    print(f"saved flow-ready checkpoint: {output}")


if __name__ == "__main__":
    main()
