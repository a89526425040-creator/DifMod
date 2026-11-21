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

Each epoch is saved atomically with the optimizer and mixed-precision scaler state. If training is interrupted, resume from the same output file:

```bash
python train_autoencoder.py --manifest /path/to/manifest.jsonl --output checkpoints/autoencoder.pt --epochs 20 --batch-size 8 --resume
```

Here `--epochs` is the total target epoch count. An intermediate checkpoint is marked as unfinished and cannot be used by flow training. Once the requested epochs finish, the script measures latent statistics and marks the checkpoint as finalized. If the process stops after the last epoch save but before finalization, rerunning with `--resume` completes that finalization step.

## 4. Train the text-conditioned flow model

```bash
python train_flow.py --manifest /path/to/manifest.jsonl --autoencoder checkpoints/autoencoder.pt --epochs 50 --batch-size 8
```

A complete checkpoint is saved atomically after every epoch, including the optimizer and mixed-precision scaler states. If a run is interrupted, repeat the command with `--resume`:

```bash
python train_flow.py --manifest /path/to/manifest.jsonl --autoencoder checkpoints/autoencoder.pt --output checkpoints/difmod_flow.pt --epochs 50 --batch-size 8 --resume
```

Here `--epochs` is the total target epoch, not the number of additional epochs. For example, if the saved checkpoint is at epoch 12 and `--epochs 50` is specified, training continues at epoch 13 and stops at 50. The resume loader checks the model configuration and latent calibration before restoring the optimizer and scaler. Keep the same autoencoder checkpoint, image size, and compatible training setup when resuming.

Flow matching is performed in normalized latent space: the saved channel statistics make the autoencoder latents approximately zero-mean and unit-scale. Sampling starts from standard Gaussian noise and converts the final latents back to the autoencoder's original scale before decoding.

This is a practical calibration baseline, not a guarantee that every latent channel is perfectly Gaussian. The statistics are estimated over the supplied training set and should be recalculated if the autoencoder or dataset changes.

The flow checkpoint includes the autoencoder, text encoder, flow transformer, latent statistics, and resume state. It is self-contained for inference and can also be resumed for training.

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

## Google Colab (Android-friendly workflow)

A ready-to-run notebook is available at [notebooks/DifMod_Training.ipynb](notebooks/DifMod_Training.ipynb). Open it in Google Colab, select a GPU runtime, mount Google Drive, and place the dataset in `MyDrive/DifModDataset` with `manifest.jsonl` and its referenced images. The notebook copies the dataset to Colab's temporary disk for faster reads and stores resumable checkpoints in `MyDrive/DifModCheckpoints`.

The notebook starts with deliberately small trial settings (2 autoencoder epochs, 5 flow epochs, batch size 2). These are pipeline smoke-test settings, not sufficient training for a useful general-purpose image generator. Increase training only after checking GPU memory, dataset quality, and autoencoder reconstructions. If the runtime disconnects, reconnect the GPU and rerun the notebook cells; existing checkpoints are resumed automatically.
