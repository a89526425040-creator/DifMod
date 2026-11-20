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

This produces `checkpoints/autoencoder.pt`. Inspect reconstruction quality before moving on. The initial autoencoder objective is deliberately simple and is not yet perceptually optimized.

## 4. Train the text-conditioned flow model

```bash
python train_flow.py --manifest /path/to/manifest.jsonl --autoencoder checkpoints/autoencoder.pt --epochs 50 --batch-size 8
```

This produces `checkpoints/difmod_flow.pt`. Training from scratch needs a sufficiently large, diverse, legally usable dataset and substantial compute. A small test dataset only verifies that the pipeline runs.

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
