"""IOB2 label definitions and validation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

LABELS: tuple[str, ...] = (
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
)
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = dict(enumerate(LABELS))


@dataclass(frozen=True)
class IOBIssue:
    """An invalid IOB2 transition, preserving its location for debugging."""

    index: int
    tag: str
    reason: str


def split_tag(tag: str) -> tuple[str, str | None]:
    """Return the IOB prefix and entity type, raising on malformed tags."""
    if tag == "O":
        return "O", None
    prefix, separator, entity_type = tag.partition("-")
    if separator != "-" or prefix not in {"B", "I"} or not entity_type:
        raise ValueError(f"Invalid IOB tag: {tag!r}")
    return prefix, entity_type


def find_iob2_issues(tags: Sequence[str]) -> list[IOBIssue]:
    """Find malformed tags and illegal I- transitions in an IOB2 sequence."""
    issues: list[IOBIssue] = []
    previous_prefix, previous_type = "O", None
    for index, tag in enumerate(tags):
        try:
            prefix, entity_type = split_tag(tag)
        except ValueError as error:
            issues.append(IOBIssue(index, tag, str(error)))
            previous_prefix, previous_type = "O", None
            continue
        if prefix == "I" and (previous_prefix == "O" or previous_type != entity_type):
            issues.append(
                IOBIssue(index, tag, "I- tag must follow B- or I- of the same entity type")
            )
        previous_prefix, previous_type = prefix, entity_type
    return issues


def normalize_iob2(tags: Iterable[str]) -> list[str]:
    """Repair illegal I- starts as B- tags while preserving entity types."""
    normalized: list[str] = []
    previous_prefix, previous_type = "O", None
    for tag in tags:
        prefix, entity_type = split_tag(tag)
        if prefix == "I" and (previous_prefix == "O" or previous_type != entity_type):
            tag, prefix = f"B-{entity_type}", "B"
        normalized.append(tag)
        previous_prefix, previous_type = prefix, entity_type
    return normalized


def is_legal_transition(previous_tag: str, current_tag: str) -> bool:
    """Whether a label pair is valid in IOB2, useful for CRF diagnostics."""
    previous_prefix, previous_type = split_tag(previous_tag)
    current_prefix, current_type = split_tag(current_tag)
    return current_prefix != "I" or (
        previous_prefix in {"B", "I"} and previous_type == current_type
    )
