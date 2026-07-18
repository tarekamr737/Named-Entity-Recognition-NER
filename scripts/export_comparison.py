"""Export a single comparable summary from EntityLens checkpoint metadata."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from entitylens.config import Paths

ARCHITECTURES = ("lstm", "bilstm", "bilstm_crf", "transformer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-samples", type=int, required=True)
    parser.add_argument("--validation-samples", type=int, required=True)
    parser.add_argument(
        "--note",
        default="Training artifacts are compared on the supplied validation split.",
    )
    return parser.parse_args()


def checkpoint_size_megabytes(path: Path) -> float:
    """Return the total saved model payload, excluding optimizer snapshots."""
    model_files = [*path.glob("*.pt"), *path.glob("*.safetensors")]
    return round(sum(file.stat().st_size for file in model_files) / (1024**2), 3)


def read_row(checkpoints: Path, architecture: str) -> dict[str, Any]:
    """Normalise the classical and Transformer metadata into one schema."""
    checkpoint = checkpoints / architecture
    metadata = json.loads((checkpoint / "training_metadata.json").read_text(encoding="utf-8"))
    validation = metadata["validation"]
    if architecture == "transformer":
        return {
            "architecture": architecture,
            "validation_f1": validation["eval_f1"],
            "precision": validation["eval_precision"],
            "recall": validation["eval_recall"],
            "runtime_seconds": metadata["training"]["train_runtime"],
            "parameter_count": metadata["parameter_count"],
            "model_size_megabytes": checkpoint_size_megabytes(checkpoint),
            "epochs": metadata["training"]["epoch"],
        }
    return {
        "architecture": architecture,
        "validation_f1": validation["f1"],
        "precision": validation["precision"],
        "recall": validation["recall"],
        "runtime_seconds": metadata["runtime_seconds"],
        "parameter_count": metadata["parameter_count"],
        "model_size_megabytes": metadata["model_size_megabytes"],
        "epochs": metadata["history"][-1]["epoch"],
    }


def main() -> None:
    args = parse_args()
    paths = Paths()
    paths.ensure()
    rows = [read_row(paths.checkpoints, architecture) for architecture in ARCHITECTURES]
    best = max(rows, key=lambda row: row["validation_f1"])
    payload = {
        "benchmark": {
            "train_samples": args.train_samples,
            "validation_samples": args.validation_samples,
            "evaluation_split": "validation",
            "test_split_used_for_training": False,
            "training_epochs": {row["architecture"]: row["epochs"] for row in rows},
            "seed": 42,
            "note": args.note,
        },
        "recommended_architecture": best["architecture"],
        "rows": rows,
    }
    json_path = paths.metrics / "model_comparison.json"
    csv_path = paths.metrics / "model_comparison.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"recommended architecture: {best['architecture']}")
    print(f"wrote {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
