# Training DifMod from scratch

## 1. Install

Use Python 3.11 and install PyTorch plus torchvision for your platform. On a CUDA machine, follow the official PyTorch selector to choose the matching CUDA wheels. On CPU:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install Pillow pytest
```

## 2. Prepare data

Create a JSONL manifest as described in [DATASET.md](DATASET.md). Keep the dataset outside this repository.

## 3. Train the image autoencoder first

```bash
python train_autoencoder.py --manifest /path/to/manifest.jsonl --epochs 20 --batch-size 8
```

The autoencoder checkpoint contains the trained encoder/decoder and per-channel latent mean and standard deviation measured across the training images. Inspect reconstruction quality before moving on. The initial autoencoder objective is deliberately simple and is not yet perceptually optimized.

## 4. Train the text-conditioned flow model

```bash
python train_flow.py --manifest /path/to/manifest.jsonl --autoencoder checkpoints/autoencoder.pt --epochs 50 --batch-size 8
```

Flow matching is performed in normalized latent space: the saved channel statistics make the autoencoder latents approximately zero-mean and unit-scale. Sampling starts from standard Gaussian noise and converts the final latents back to the autoencoder's original scale before decoding.

This is a practical calibration baseline, not a guarantee that every latent channel is perfectly Gaussian. The statistics are estimated over the supplied training set and should be recalculated if the autoencoder or dataset changes.

The flow checkpoint includes the autoencoder, text encoder, flow transformer, and latent statistics. This makes inference self-contained once a checkpoint has been trained.

## 5. Measure model size

```bash
python -m scripts.count_parameters
```

## Current limitations

- The byte tokenizer is a baseline and does not understand semantics by itself; it must learn text/image associations during training.
- The autoencoder is deterministic, not a pretrained or perceptually trained VAE.
- The first flow model is a research prototype. It has no quality guarantee.
- Training scripts currently target a workstation/cloud PyTorch environment, not on-device training.
- Android export, quantization, and Snapdragon benchmarks are later milestones.
