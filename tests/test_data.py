import json

import torch
from PIL import Image

from difmod.data import ImageCaptionDataset


def test_manifest_dataset_loads_rgb_image(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (40, 30), (120, 30, 20)).save(image_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"image": "sample.png", "caption": "a red object"}) + "\n",
                        encoding="utf-8")
    dataset = ImageCaptionDataset(manifest, image_size=32)
    image, caption = dataset[0]
    assert image.shape == (3, 32, 32)
    assert torch.isfinite(image).all()
    assert caption == "a red object"
