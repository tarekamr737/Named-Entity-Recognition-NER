"""Artifact-backed inference helpers for the EntityLens Streamlit workbench."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from .data import Vocabulary
from .labels import ID_TO_LABEL, normalize_iob2
from .models import create_classical_model

TOKEN_PATTERN = re.compile(r"\w+(?:['-]\w+)*|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class PredictedEntity:
    """One text span produced by an EntityLens inference backend."""

    text: str
    entity_type: str
    confidence: float
    start: int
    end: int


@dataclass(frozen=True)
class InferenceResult:
    """Structured result that supports rendering, tables, and diagnostics."""

    entities: list[PredictedEntity]
    token_tags: list[tuple[str, str, float]]
    elapsed_ms: float
    source: str


def available_architectures(checkpoint_root: Path) -> list[str]:
    """Return only model artifacts that can be selected in the UI."""
    candidates = ["transformer", "bilstm_crf", "bilstm", "lstm"]
    return [name for name in candidates if (checkpoint_root / name).is_dir()]


def load_classical_model(checkpoint_dir: Path, architecture: str) -> tuple[Any, Vocabulary]:
    """Load a classical checkpoint and its exact vocabulary without retraining."""
    checkpoint = torch.load(checkpoint_dir / "model.pt", map_location="cpu", weights_only=True)
    with (checkpoint_dir / "word_vocab.json").open(encoding="utf-8") as handle:
        vocabulary = Vocabulary(**json.load(handle))
    model = create_classical_model(architecture, **checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, vocabulary


def _token_spans(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(), match.start(), match.end()) for match in TOKEN_PATTERN.finditer(text)]


def _entities_from_iob(
    text: str, tokens: list[tuple[str, int, int]], tags: list[str], confidences: list[float]
) -> list[PredictedEntity]:
    entities: list[PredictedEntity] = []
    current_start: int | None = None
    current_end: int | None = None
    current_type: str | None = None
    scores: list[float] = []

    def close_entity() -> None:
        nonlocal current_start, current_end, current_type, scores
        if current_start is not None and current_end is not None and current_type is not None:
            entities.append(
                PredictedEntity(
                    text=text[current_start:current_end],
                    entity_type=current_type,
                    confidence=round(sum(scores) / len(scores), 3),
                    start=current_start,
                    end=current_end,
                )
            )
        current_start = current_end = None
        current_type = None
        scores = []

    for (_, start, end), tag, score in zip(tokens, normalize_iob2(tags), confidences, strict=True):
        prefix, _, entity_type = tag.partition("-")
        is_continuation = prefix == "I" and entity_type == current_type
        if prefix == "O" or not is_continuation and prefix != "B":
            close_entity()
            continue
        if prefix == "B" or entity_type != current_type:
            close_entity()
            current_start, current_type = start, entity_type
        current_end = end
        scores.append(score)
    close_entity()
    return entities


@torch.inference_mode()
def predict_classical(text: str, model: Any, vocabulary: Vocabulary) -> InferenceResult:
    """Run word-level model inference and reassemble IOB labels into spans."""
    started = perf_counter()
    tokens = _token_spans(text)
    if not tokens:
        return InferenceResult([], [], 0.0, "checkpoint")
    words = [token for token, _, _ in tokens]
    input_ids = torch.tensor([vocabulary.encode(words)], dtype=torch.long)
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    logits = model(input_ids, mask)
    probabilities = torch.softmax(logits, dim=-1)[0]
    decoded = model.decode(input_ids, mask)[0]
    tags = [ID_TO_LABEL[index] for index in decoded]
    scores = [float(probabilities[index, label_id]) for index, label_id in enumerate(decoded)]
    return InferenceResult(
        _entities_from_iob(text, tokens, tags, scores),
        list(zip(words, tags, scores, strict=True)),
        round((perf_counter() - started) * 1000, 1),
        "checkpoint",
    )


def load_transformer_model(checkpoint_dir: Path) -> tuple[Any, Any]:
    """Load the locally saved transformer without a remote model lookup."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir, local_files_only=True)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def predict_transformer(text: str, tokenizer: Any, model: Any) -> InferenceResult:
    """Run local transformer inference and merge adjacent compatible subwords."""
    started = perf_counter()
    if not text.strip():
        return InferenceResult([], [], 0.0, "checkpoint")
    encoded = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True)
    offsets = encoded.pop("offset_mapping")[0].tolist()
    output = model(**encoded).logits[0]
    probabilities = torch.softmax(output, dim=-1)
    label_ids = probabilities.argmax(dim=-1).tolist()
    labels = [model.config.id2label[index] for index in label_ids]
    entities: list[PredictedEntity] = []
    current: PredictedEntity | None = None
    for index, ((start, end), label, label_id) in enumerate(
        zip(offsets, labels, label_ids, strict=True)
    ):
        if start == end or label == "O":
            if current:
                entities.append(current)
                current = None
            continue
        prefix, _, entity_type = label.partition("-")
        confidence = float(probabilities[index, label_id])
        can_join = (
            current
            and prefix == "I"
            and current.entity_type == entity_type
            and start <= current.end
        )
        if can_join:
            new_end = max(current.end, end)
            current = PredictedEntity(
                text=text[current.start:new_end],
                entity_type=entity_type,
                confidence=round((current.confidence + confidence) / 2, 3),
                start=current.start,
                end=new_end,
            )
        else:
            if current:
                entities.append(current)
            current = PredictedEntity(
                text=text[start:end],
                entity_type=entity_type,
                confidence=round(confidence, 3),
                start=start,
                end=end,
            )
    if current:
        entities.append(current)
    visible_tokens = [token for token, _, _ in _token_spans(text)]
    return InferenceResult(
        entities,
        [(token, "O", 1.0) for token in visible_tokens],
        round((perf_counter() - started) * 1000, 1),
        "checkpoint",
    )


def rule_fallback(text: str) -> InferenceResult:
    """Provide a clearly labelled local fallback when trained weights cannot load."""
    started = perf_counter()
    organisations = {"Microsoft", "Google", "OpenAI", "Apple", "IBM", "United Nations"}
    locations = {"Cairo", "London", "Seattle", "Zurich", "Paris", "Egypt", "Europe"}
    entities: list[PredictedEntity] = []
    for token, start, end in _token_spans(text):
        entity_type = "ORG" if token in organisations else "LOC" if token in locations else None
        if entity_type:
            entities.append(PredictedEntity(token, entity_type, 0.62, start, end))
    return InferenceResult(
        entities,
        [(token, "O", 1.0) for token, _, _ in _token_spans(text)],
        round((perf_counter() - started) * 1000, 1),
        "fallback",
    )
