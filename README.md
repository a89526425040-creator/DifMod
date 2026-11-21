# DifMod — mobile-first image generation research

DifMod is an experimental, independently implemented text-to-image model designed around the memory and compute limits of Android phones such as devices powered by Snapdragon 7s Gen 3.

**Important:** this repository starts with an untrained research architecture. It does not contain pretrained weights and cannot produce meaningful prompt-following images until it has been trained on a properly licensed image-caption dataset.

## Design goals

- Train a small model from scratch; no Stable Diffusion, FLUX, Krea, or other pretrained generator weights.
- Keep the model modular: UTF-8 byte text encoder, compact latent image autoencoder, and conditional flow-matching transformer.
- Begin with small training/inference resolutions, then measure quality, latency, memory use, and thermals before scaling.
- Export and quantize only after the FP32/FP16 model is numerically validated.
- Treat Android GPU/NPU acceleration as a later measured integration, not an assumption.

## Initial research configuration

The first prototype uses 256×256 training images, an 8× spatial latent compression target, a compact transformer, and a byte-level text tokenizer. These are experimental defaults, not a claim that the first model will already generate high-quality images or run at a particular speed on a phone.

## Repository layout

- `difmod/` — model, tokenizer, flow-matching objective, inference, and training utilities.
- `tests/` — CPU shape, numerical, and checkpoint smoke tests.
- `configs/` — small starter configuration.
- `.github/workflows/` — automated tests.
- `generate_image.py` — command-line image generation from a trained checkpoint.

## Development

Python 3.11 is recommended.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

## Generate an image

After training has produced the complete checkpoint `checkpoints/difmod_flow.pt`, run:

```bash
python generate_image.py --checkpoint checkpoints/difmod_flow.pt --prompt "a small red car on a country road" --output result.png --steps 20
```

The script loads the autoencoder, text encoder, flow transformer, and calibrated latent statistics from the unified checkpoint. It saves a PNG image. An untrained or randomly initialized checkpoint will only produce noise-like output; the command does not replace the training step.

## Training data

Use image-caption pairs that you own or are licensed to use. Keep the original dataset outside Git; store only a manifest and documented preparation instructions in this repository. Never commit private images or large datasets.

## Roadmap

1. Establish a tested, trainable architecture.
2. Add a licensed image-caption dataset loader and training loop.
3. Measure parameter count, peak memory, and throughput on available hardware.
4. Train a first checkpoint on cloud compute.
5. Benchmark and optimize Android inference (CPU first, then supported GPU delegates).
6. Build the Android application.

## Status

Prototype / research. No trained checkpoint is included.
