"""Entity-level evaluation shared across classical and transformer models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

from .labels import ID_TO_LABEL

ENTITY_TYPES = ("PER", "ORG", "LOC", "MISC")


@dataclass(frozen=True)
class EntityMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class EvaluationResult:
    precision: float
    recall: float
    f1: float
    per_entity: dict[str, EntityMetrics]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["per_entity"] = {
            entity: asdict(metrics) for entity, metrics in self.per_entity.items()
        }
        return result


@dataclass(frozen=True)
class BoundaryErrorSummary:
    """Exact span and boundary-overlap counts for a predicted tag sequence."""

    true_entities: int
    predicted_entities: int
    exact_matches: int
    boundary_errors: int
    missed_entities: int
    spurious_entities: int

    @property
    def complete_entity_accuracy(self) -> float:
        """The share of gold spans recovered with type and boundaries intact."""
        return round(self.exact_matches / self.true_entities, 4) if self.true_entities else 0.0

    def to_dict(self) -> dict[str, int | float]:
        return {**asdict(self), "complete_entity_accuracy": self.complete_entity_accuracy}


def entity_spans(tags: Sequence[str]) -> list[tuple[int, int, str]]:
    """Convert a normalised IOB2 tag sequence into token-indexed entity spans."""
    spans: list[tuple[int, int, str]] = []
    start: int | None = None
    entity_type: str | None = None
    for index, tag in enumerate([*tags, "O"]):
        prefix, _, tag_type = tag.partition("-")
        if prefix == "I" and tag_type == entity_type:
            continue
        if start is not None and entity_type is not None:
            spans.append((start, index, entity_type))
            start, entity_type = None, None
        if prefix == "B":
            start, entity_type = index, tag_type
    return spans


def boundary_error_summary(
    true_tags: Sequence[str], predicted_tags: Sequence[str]
) -> BoundaryErrorSummary:
    """Separate exact matches from overlapping, wrong-boundary entity predictions."""
    true_spans = entity_spans(true_tags)
    predicted_spans = entity_spans(predicted_tags)
    exact = set(true_spans) & set(predicted_spans)
    unmatched_true = [span for span in true_spans if span not in exact]
    unmatched_predicted = [span for span in predicted_spans if span not in exact]
    boundary_errors = 0
    matched_true_indices: set[int] = set()
    for predicted in unmatched_predicted:
        predicted_start, predicted_end, predicted_type = predicted
        for index, truth in enumerate(unmatched_true):
            true_start, true_end, true_type = truth
            overlaps = max(predicted_start, true_start) < min(predicted_end, true_end)
            if index not in matched_true_indices and predicted_type == true_type and overlaps:
                matched_true_indices.add(index)
                boundary_errors += 1
                break
    return BoundaryErrorSummary(
        true_entities=len(true_spans),
        predicted_entities=len(predicted_spans),
        exact_matches=len(exact),
        boundary_errors=boundary_errors,
        missed_entities=len(unmatched_true) - boundary_errors,
        spurious_entities=len(unmatched_predicted) - boundary_errors,
    )


def label_sequences_from_ids(
    predictions: Sequence[Sequence[int]],
    targets: Sequence[Sequence[int]],
    *,
    ignored_label: int = -100,
) -> tuple[list[list[str]], list[list[str]]]:
    """Convert numeric sequences to seqeval inputs, ignoring padded targets."""
    true_sequences: list[list[str]] = []
    predicted_sequences: list[list[str]] = []
    for predicted, target in zip(predictions, targets, strict=True):
        true_tags: list[str] = []
        predicted_tags: list[str] = []
        for predicted_id, target_id in zip(predicted, target, strict=False):
            if target_id == ignored_label:
                continue
            true_tags.append(ID_TO_LABEL[int(target_id)])
            predicted_tags.append(ID_TO_LABEL[int(predicted_id)])
        true_sequences.append(true_tags)
        predicted_sequences.append(predicted_tags)
    return true_sequences, predicted_sequences


def evaluate_sequences(
    true_sequences: Sequence[Sequence[str]], predicted_sequences: Sequence[Sequence[str]]
) -> EvaluationResult:
    """Compute strict entity spans overall and separately for all CoNLL types."""
    report = classification_report(
        true_sequences, predicted_sequences, output_dict=True, zero_division=0
    )
    per_entity = {
        entity: EntityMetrics(
            precision=round(float(report.get(entity, {}).get("precision", 0.0)), 4),
            recall=round(float(report.get(entity, {}).get("recall", 0.0)), 4),
            f1=round(float(report.get(entity, {}).get("f1-score", 0.0)), 4),
            support=int(report.get(entity, {}).get("support", 0)),
        )
        for entity in ENTITY_TYPES
    }
    return EvaluationResult(
        precision=round(
            float(precision_score(true_sequences, predicted_sequences, zero_division=0)), 4
        ),
        recall=round(float(recall_score(true_sequences, predicted_sequences, zero_division=0)), 4),
        f1=round(float(f1_score(true_sequences, predicted_sequences, zero_division=0)), 4),
        per_entity=per_entity,
    )
