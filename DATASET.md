# Preparing a training dataset

DifMod learns from image-caption pairs. Use only images and captions you own or are licensed to use.

Create a JSONL manifest. Each line is one JSON object:

```json
{"image": "images/car_001.jpg", "caption": "a red sports car parked beside a lake"}
{"image": "images/house_001.png", "caption": "a small wooden house in a snowy forest"}
```

Paths are relative to the manifest file unless absolute. Keep the image files and manifest outside Git; do not commit private images, large datasets, or copyrighted collections without permission.

Start with a small, diverse, carefully captioned dataset to validate the pipeline. A tiny dataset is useful for debugging but will not produce a general-purpose image generator.
