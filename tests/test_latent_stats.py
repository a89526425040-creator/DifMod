import torch
from torch.utils.data import DataLoader, Dataset

from difmod.autoencoder import LatentAutoencoder
from difmod.latent_stats import estimate_latent_stats


class TinyImageDataset(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        return torch.rand(3, 32, 32, generator=generator) * 2 - 1, "sample"


def test_latent_statistics_are_finite_and_per_channel():
    autoencoder = LatentAutoencoder(latent_channels=4, base_channels=8)
    loader = DataLoader(TinyImageDataset(), batch_size=2)
    mean, std = estimate_latent_stats(autoencoder, loader, "cpu")
    assert mean.shape == (1, 4, 1, 1)
    assert std.shape == (1, 4, 1, 1)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(std).all()
    assert (std > 0).all()
