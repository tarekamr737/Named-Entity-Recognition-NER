"""Compare exact spans, boundary errors, and IOB validity for classical checkpoints."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from entitylens.config import Paths
from entitylens.data import Vocabulary, load_conll2003
from entitylens.evaluation import boundary_error_summary
from entitylens.inference import load_classical_model
from entitylens.labels import ID_TO_LABEL, find_iob2_issues

ARCHITECTURES = ("lstm", "bilstm", "bilstm_crf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-samples", type=int, default=32)
    return parser.parse_args()


def predict_tags(model: Any, vocabulary: Vocabulary, tokens: list[str]) -> list[str]:
    """Decode a tokenised sentence exactly as the classical trainer does."""
    input_ids = torch.tensor([vocabulary.encode(tokens)], dtype=torch.long)
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    return [ID_TO_LABEL[label_id] for label_id in model.decode(input_ids, mask)[0]]


def analyse_architecture(
    rows: list[dict[str, Any]], checkpoint: Path, architecture: str
) -> dict[str, Any]:
    """Aggregate error categories with one representative example per error type."""
    model, vocabulary = load_classical_model(checkpoint, architecture)
    totals: dict[str, int | float] = {
        "true_entities": 0,
        "predicted_entities": 0,
        "exact_matches": 0,
        "boundary_errors": 0,
        "missed_entities": 0,
        "spurious_entities": 0,
        "invalid_iob2_transitions": 0,
    }
    examples: dict[str, dict[str, Any]] = {}
    confusions: Counter[tuple[str, str]] = Counter()
    for row in rows:
        true_tags = [ID_TO_LABEL[int(tag)] for tag in row["ner_tags"]]
        predicted_tags = predict_tags(model, vocabulary, list(row["tokens"]))
        summary = boundary_error_summary(true_tags, predicted_tags).to_dict()
        confusions.update(
            (truth, predicted)
            for truth, predicted in zip(true_tags, predicted_tags, strict=True)
            if truth != predicted
        )
        for key, value in summary.items():
            if key != "complete_entity_accuracy":
                totals[key] += value
        totals["invalid_iob2_transitions"] += len(find_iob2_issues(predicted_tags))
        for key in ("exact_matches", "boundary_errors", "missed_entities", "spurious_entities"):
            if summary[key] and key not in examples:
                examples[key] = {
                    "tokens": row["tokens"],
                    "true_tags": true_tags,
                    "predicted_tags": predicted_tags,
                }
    totals["complete_entity_accuracy"] = (
        round(totals["exact_matches"] / totals["true_entities"], 4)
        if totals["true_entities"]
        else 0.0
    )
    return {
        "summary": totals,
        "examples": examples,
        "top_label_confusions": [
            {"true": truth, "predicted": predicted, "count": count}
            for (truth, predicted), count in confusions.most_common(5)
        ],
    }


def main() -> None:
    args = parse_args()
    paths = Paths()
    paths.ensure()
    dataset = load_conll2003(paths.root / "conll2003.py", paths.data_raw)
    rows = list(dataset["validation"].select(range(args.validation_samples)))
    analyses = {
        architecture: analyse_architecture(rows, paths.checkpoints / architecture, architecture)
        for architecture in ARCHITECTURES
    }
    crf = analyses["bilstm_crf"]["summary"]
    bilstm = analyses["bilstm"]["summary"]
    payload = {
        "validation_samples": args.validation_samples,
        "architectures": analyses,
        "bilstm_to_crf_delta": {
            "complete_entity_accuracy": round(
                crf["complete_entity_accuracy"] - bilstm["complete_entity_accuracy"], 4
            ),
            "boundary_errors": crf["boundary_errors"] - bilstm["boundary_errors"],
            "invalid_iob2_transitions": (
                crf["invalid_iob2_transitions"] - bilstm["invalid_iob2_transitions"]
            ),
        },
    }
    output = paths.metrics / "boundary_analysis.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
