"""Vocabulary-aligned loading for plain-text GloVe vector files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .data import Vocabulary


@dataclass(frozen=True)
class EmbeddingLoadReport:
    """Embedding matrix and coverage facts persisted with training metadata."""

    weights: Tensor
    source: str
    dimension: int
    matched_tokens: int
    vocabulary_tokens: int

    @property
    def coverage(self) -> float:
        if not self.vocabulary_tokens:
            return 0.0
        return round(self.matched_tokens / self.vocabulary_tokens, 4)

    def metadata(self) -> dict[str, str | int | float]:
        return {
            "source": self.source,
            "dimension": self.dimension,
            "matched_tokens": self.matched_tokens,
            "vocabulary_tokens": self.vocabulary_tokens,
            "coverage": self.coverage,
        }


def load_glove_embeddings(path: str | Path, vocabulary: Vocabulary) -> EmbeddingLoadReport:
    """Initialise a vocabulary matrix from a standard space-delimited GloVe file."""
    vector_path = Path(path)
    if not vector_path.is_file():
        raise FileNotFoundError(f"GloVe vector file not found: {vector_path}")
    dimension: int | None = None
    matched: dict[int, Tensor] = {}
    with vector_path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip().split(" ")
            if len(parts) < 3:
                continue
            token = parts[0].lower() if vocabulary.lowercase else parts[0]
            token_id = vocabulary.token_to_id.get(token)
            if token_id is None:
                continue
            try:
                vector = torch.tensor([float(value) for value in parts[1:]], dtype=torch.float32)
            except ValueError:
                continue
            if dimension is None:
                dimension = vector.numel()
            if vector.numel() == dimension:
                matched[token_id] = vector
    if dimension is None:
        raise ValueError(f"No valid vectors found in {vector_path}")
    weights = torch.empty((len(vocabulary.token_to_id), dimension), dtype=torch.float32)
    torch.nn.init.normal_(weights, mean=0.0, std=0.05)
    for token_id, vector in matched.items():
        weights[token_id] = vector
    weights[vocabulary.pad_id].zero_()
    vocabulary_tokens = len(vocabulary.token_to_id) - 2
    return EmbeddingLoadReport(
        weights=weights,
        source=str(vector_path),
        dimension=dimension,
        matched_tokens=len(matched),
        vocabulary_tokens=vocabulary_tokens,
    )
