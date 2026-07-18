"""Fine-tune DistilBERT or BERT on the shared EntityLens CoNLL-2003 split."""

from __future__ import annotations

import argparse

from entitylens.config import Paths
from entitylens.data import load_conll2003
from entitylens.reproducibility import configure_logging, set_seed
from entitylens.transformer_training import train_transformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="distilbert/distilbert-base-cased")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-validation", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    set_seed(args.seed)
    paths = Paths()
    paths.ensure()
    dataset = load_conll2003(paths.root / "conll2003.py", paths.data_raw)
    if args.max_train:
        train_count = min(args.max_train, len(dataset["train"]))
        dataset["train"] = dataset["train"].select(range(train_count))
    if args.max_validation:
        dataset["validation"] = dataset["validation"].select(
            range(min(args.max_validation, len(dataset["validation"])))
        )
    metadata = train_transformer(
        dataset,
        paths.checkpoints / "transformer",
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    print(f"transformer validation F1: {metadata['validation'].get('eval_f1', 0.0):.4f}")


if __name__ == "__main__":
    main()
