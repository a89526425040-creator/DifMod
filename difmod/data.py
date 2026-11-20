from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageCaptionDataset(Dataset):
    """Dataset described by a JSONL manifest.

    Each line: {"image": "relative/or/absolute/path.png", "caption": "a red car"}
    Paths are resolved relative to the manifest's parent directory.
    """

    def __init__(self, manifest: str | Path, image_size: int = 256) -> None:
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.items: list[dict[str, str]] = []
        with self.manifest.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if "image" not in item or "caption" not in item:
                    raise ValueError(f"manifest line {line_number} needs image and caption")
                self.items.append({"image": str(item["image"]), "caption": str(item["caption"])})
        if not self.items:
            raise ValueError("dataset manifest contains no examples")
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        item = self.items[index]
        path = Path(item["image"])
        if not path.is_absolute():
            path = self.root / path
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, item["caption"]


def collate_images_and_captions(batch):
    images, captions = zip(*batch)
    return torch.stack(images), list(captions)
