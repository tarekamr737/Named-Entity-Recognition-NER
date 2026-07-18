"""Reproducible training and checkpointing for the classical NER models."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .data import Vocabulary, pad_sequences
from .evaluation import EvaluationResult, evaluate_sequences, label_sequences_from_ids
from .labels import LABEL_TO_ID
from .models import ArchitectureName, BiLSTMCRFTokenClassifier, LSTMTokenClassifier
from .reproducibility import set_seed


def label_ids(tags: Iterable[int | str]) -> list[int]:
    """Convert a CoNLL label sequence from ids or IOB strings into label ids."""
    result: list[int] = []
    for tag in tags:
        if isinstance(tag, int):
            result.append(tag)
        else:
            result.append(LABEL_TO_ID[tag])
    return result


class TokenDataset(Dataset[tuple[list[int], list[int]]]):
    """Word-id and tag-id examples with no padding stored in the dataset."""

    def __init__(self, rows: Sequence[dict[str, Any]], vocabulary: Vocabulary) -> None:
        self.examples = [
            (vocabulary.encode(row["tokens"]), label_ids(row["ner_tags"])) for row in rows
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return self.examples[index]


def collate_token_batch(
    batch: Sequence[tuple[list[int], list[int]]], pad_id: int = 0
) -> dict[str, Tensor]:
    """Dynamically pad a batch and retain the attention mask for every model."""
    token_sequences, label_sequences = zip(*batch, strict=True)
    input_ids, masks = pad_sequences(token_sequences, pad_id)
    labels, _ = pad_sequences(label_sequences, -100, max_length=len(input_ids[0]))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.bool),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


@dataclass(frozen=True)
class TrainingResult:
    architecture: str
    validation: EvaluationResult
    history: list[dict[str, float]]
    runtime_seconds: float
    parameter_count: int
    model_size_megabytes: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "validation": self.validation.to_dict(),
            "history": self.history,
            "runtime_seconds": round(self.runtime_seconds, 3),
            "parameter_count": self.parameter_count,
            "model_size_megabytes": round(self.model_size_megabytes, 3),
        }


Model = LSTMTokenClassifier | BiLSTMCRFTokenClassifier


@torch.inference_mode()
def evaluate_classical(
    model: Model, loader: DataLoader[dict[str, Tensor]], device: torch.device
) -> EvaluationResult:
    """Decode a model and calculate strict entity metrics on a data loader."""
    model.eval()
    predictions: list[list[int]] = []
    targets: list[list[int]] = []
    for batch in loader:
        inputs = batch["input_ids"].to(device)
        masks = batch["attention_mask"].to(device)
        decoded = model.decode(inputs, masks)
        predictions.extend(decoded)
        targets.extend(batch["labels"].tolist())
    true_sequences, predicted_sequences = label_sequences_from_ids(predictions, targets)
    return evaluate_sequences(true_sequences, predicted_sequences)


def train_classical(
    model: Model,
    train_rows: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
    vocabulary: Vocabulary,
    *,
    architecture: ArchitectureName,
    batch_size: int = 32,
    epochs: int = 8,
    learning_rate: float = 3e-4,
    seed: int = 42,
    device: str | None = None,
) -> TrainingResult:
    """Train a classical model with shared splits, optimizer, and evaluation."""
    set_seed(seed)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(target_device)
    train_loader = DataLoader(
        TokenDataset(train_rows, vocabulary),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_token_batch(batch, vocabulary.pad_id),
    )
    validation_loader = DataLoader(
        TokenDataset(validation_rows, vocabulary),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_token_batch(batch, vocabulary.pad_id),
    )
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(
                batch["input_ids"].to(target_device),
                batch["labels"].to(target_device),
                batch["attention_mask"].to(target_device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())
        validation = evaluate_classical(model, validation_loader, target_device)
        history.append(
            {
                "epoch": float(epoch),
                "loss": round(total_loss / max(len(train_loader), 1), 4),
                "validation_f1": validation.f1,
            }
        )
    runtime_seconds = time.perf_counter() - started
    final_validation = evaluate_classical(model, validation_loader, target_device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    size_bytes = sum(
        parameter.nelement() * parameter.element_size() for parameter in model.parameters()
    )
    return TrainingResult(
        architecture=architecture,
        validation=final_validation,
        history=history,
        runtime_seconds=runtime_seconds,
        parameter_count=parameter_count,
        model_size_megabytes=size_bytes / (1024**2),
    )


def save_classical_artifacts(
    model: Model,
    result: TrainingResult,
    vocabulary: Vocabulary,
    output_dir: str | Path,
    *,
    model_config: dict[str, Any],
) -> None:
    """Persist a loadable checkpoint, vocabulary, labels, and training report."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "model_config": model_config},
        destination / "model.pt",
    )
    (destination / "word_vocab.json").write_text(
        json.dumps(vocabulary.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (destination / "label_map.json").write_text(
        json.dumps(LABEL_TO_ID, indent=2, sort_keys=True), encoding="utf-8"
    )
    (destination / "training_metadata.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
