from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from difmod.inference import generate_images, load_trained_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one image with a trained DifMod checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="checkpoint produced by train_flow.py")
    parser.add_argument("--prompt", required=True, help="text description of the image")
    parser.add_argument("--output", default="difmod_output.png")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be at least 1")
    device = torch.device(args.device)
    model = load_trained_model(args.checkpoint, device=device)
    image = generate_images(
        model, [args.prompt], steps=args.steps, seed=args.seed, device=device
    )[0]
    pixels = ((image.cpu().clamp(-1, 1) + 1.0) * 127.5).round().byte()
    pixels = pixels.permute(1, 2, 0).numpy()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(output)
    print(f"saved image: {output}")


if __name__ == "__main__":
    main()
