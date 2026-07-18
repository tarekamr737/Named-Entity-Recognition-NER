"""Train one LSTM-family EntityLens model on the shared CoNLL-2003 split."""

from __future__ import annotations

import argparse
from pathlib import Path

from entitylens.config import Paths, TrainingConfig
from entitylens.data import build_vocabulary, load_conll2003
from entitylens.embeddings import load_glove_embeddings
from entitylens.labels import LABELS
from entitylens.models import create_classical_model
from entitylens.reproducibility import configure_logging
from entitylens.training import save_classical_artifacts, train_classical


def parse_args() -> argparse.Namespace:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("architecture", choices=["lstm", "bilstm", "bilstm_crf"])
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-validation", type=int, default=None)
    parser.add_argument("--glove-file", type=Path, default=None)
    parser.add_argument("--checkpoint-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    paths = Paths()
    paths.ensure()
    dataset = load_conll2003(paths.root / "conll2003.py", paths.data_raw)
    train_rows = list(dataset["train"])
    validation_rows = list(dataset["validation"])
    if args.max_train:
        train_rows = train_rows[: args.max_train]
    if args.max_validation:
        validation_rows = validation_rows[: args.max_validation]
    vocabulary = build_vocabulary([row["tokens"] for row in train_rows])
    defaults = TrainingConfig()
    embedding_report = (
        load_glove_embeddings(args.glove_file, vocabulary) if args.glove_file else None
    )
    model_config = {
        "vocab_size": len(vocabulary.token_to_id),
        "num_labels": len(LABELS),
        "embedding_dim": embedding_report.dimension if embedding_report else defaults.embedding_dim,
        "hidden_dim": defaults.hidden_dim,
        "dropout": defaults.dropout,
        "pad_id": vocabulary.pad_id,
    }
    model = create_classical_model(
        args.architecture,
        **model_config,
        **({"embedding_weights": embedding_report.weights} if embedding_report else {}),
    )
    result = train_classical(
        model,
        train_rows,
        validation_rows,
        vocabulary,
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    destination = paths.checkpoints / (args.checkpoint_name or args.architecture)
    save_classical_artifacts(model, result, vocabulary, destination, model_config=model_config)
    if embedding_report:
        metadata_path = destination / "training_metadata.json"
        import json

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["pretrained_embeddings"] = embedding_report.metadata()
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        print(f"GloVe coverage: {embedding_report.coverage:.1%}")
    print(f"{args.architecture} validation F1: {result.validation.f1:.4f}")


if __name__ == "__main__":
    main()
