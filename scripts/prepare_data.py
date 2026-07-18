"""Download, validate, and profile CoNLL-2003 for EntityLens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from entitylens.config import Paths
from entitylens.data import (
    build_character_vocabulary,
    build_vocabulary,
    dataset_diagnostics,
    load_conll2003,
)
from entitylens.labels import LABEL_TO_ID
from entitylens.reproducibility import configure_logging, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-script",
        type=Path,
        default=Paths().root / "conll2003.py",
        help="Path to the provided Hugging Face CoNLL-2003 builder.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging()
    set_seed(args.seed)
    paths = Paths()
    paths.ensure()

    dataset = load_conll2003(args.dataset_script, paths.data_raw)
    diagnostics = dataset_diagnostics(dataset)
    train_tokens = dataset["train"]["tokens"]
    word_vocab = build_vocabulary(train_tokens)
    char_vocab = build_character_vocabulary(train_tokens)

    write_json(paths.metrics / "dataset_diagnostics.json", diagnostics.to_dict())
    write_json(paths.vocabularies / "word_vocab.json", word_vocab.to_dict())
    write_json(paths.vocabularies / "char_vocab.json", char_vocab.to_dict())
    write_json(paths.vocabularies / "label_to_id.json", LABEL_TO_ID)

    print(
        "Prepared CoNLL-2003: "
        f"{diagnostics.split_sizes}, {len(word_vocab.token_to_id)} word tokens, "
        f"{len(char_vocab.token_to_id)} characters."
    )


if __name__ == "__main__":
    main()
