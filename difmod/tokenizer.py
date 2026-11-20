from __future__ import annotations

import torch


class ByteTokenizer:
    """Tiny UTF-8 byte tokenizer: PAD=0, bytes=1..256, BOS=257, EOS=258.

    This deliberately avoids downloading a pretrained tokenizer. It is compact
    and multilingual at the byte level, but less semantically efficient than a
    learned subword tokenizer.
    """

    pad_id = 0
    bos_id = 257
    eos_id = 258
    vocab_size = 259

    def __init__(self, max_length: int = 128) -> None:
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        self.max_length = max_length

    def encode(self, text: str) -> list[int]:
        payload = list(text.encode("utf-8"))[: self.max_length - 2]
        return [self.bos_id, *[value + 1 for value in payload], self.eos_id]

    def batch_encode(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if not texts:
            raise ValueError("texts must not be empty")
        encoded = [self.encode(text) for text in texts]
        width = max(len(row) for row in encoded)
        tokens = torch.full((len(encoded), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(encoded), width), dtype=torch.bool)
        for index, row in enumerate(encoded):
            tokens[index, : len(row)] = torch.tensor(row, dtype=torch.long)
            mask[index, : len(row)] = True
        return tokens, mask
