"""CoNLL-2003 loading, diagnostics, vocabularies, and transformer alignment."""

from __future__ import annotations

import importlib.util
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .labels import LABELS, find_iob2_issues, normalize_iob2

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_LABEL_ID = -100


@dataclass(frozen=True)
class Vocabulary:
    """A serializable vocabulary with explicit padding and unknown values."""

    token_to_id: dict[str, int]
    lowercase: bool = True

    @property
    def id_to_token(self) -> dict[int, str]:
        return {index: token for token, index in self.token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    def encode(self, tokens: Iterable[str]) -> list[int]:
        return [
            self.token_to_id.get(token.lower() if self.lowercase else token, self.unk_id)
            for token in tokens
        ]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetDiagnostics:
    """Small, JSON-ready summary used in reports and training metadata."""

    split_sizes: dict[str, int]
    entity_frequencies: dict[str, int]
    label_frequencies: dict[str, int]
    sentence_length: dict[str, float]
    invalid_iob2_tags: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_vocabulary(
    sequences: Iterable[Sequence[str]], *, lowercase: bool = True, min_frequency: int = 1
) -> Vocabulary:
    """Build a deterministic word or character vocabulary from training data only."""
    counts: Counter[str] = Counter()
    for sequence in sequences:
        counts.update(token.lower() if lowercase else token for token in sequence)
    vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token in sorted(counts):
        if counts[token] >= min_frequency:
            vocabulary[token] = len(vocabulary)
    return Vocabulary(vocabulary, lowercase=lowercase)


def build_character_vocabulary(tokens: Iterable[Sequence[str]]) -> Vocabulary:
    """Build a character vocabulary from tokenized training sentences."""
    return build_vocabulary((list("".join(sentence)) for sentence in tokens), lowercase=False)


def pad_sequences(
    sequences: Sequence[Sequence[int]], pad_value: int, max_length: int | None = None
) -> tuple[list[list[int]], list[list[int]]]:
    """Pad or truncate integer sequences and return matching binary attention masks."""
    if not sequences:
        return [], []
    target_length = max_length or max(len(sequence) for sequence in sequences)
    padded: list[list[int]] = []
    masks: list[list[int]] = []
    for sequence in sequences:
        values = list(sequence[:target_length])
        padding = target_length - len(values)
        padded.append(values + [pad_value] * padding)
        masks.append([1] * len(values) + [0] * padding)
    return padded, masks


def align_token_labels(
    tokenized_batch: Any,
    word_labels: Sequence[Sequence[int]],
    *,
    label_all_subtokens: bool = False,
) -> list[list[int]]:
    """Align word-level labels to a fast tokenizer's subword positions.

    Special tokens and non-leading subtokens are ignored by default with -100,
    which is the loss ignore value expected by Hugging Face token classifiers.
    """
    aligned: list[list[int]] = []
    for batch_index, labels in enumerate(word_labels):
        word_ids = tokenized_batch.word_ids(batch_index=batch_index)
        previous_word_id: int | None = None
        aligned_labels: list[int] = []
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(PAD_LABEL_ID)
            elif word_id != previous_word_id:
                aligned_labels.append(labels[word_id])
            elif label_all_subtokens:
                label = labels[word_id]
                aligned_labels.append(label + 1 if label % 2 == 1 else label)
            else:
                aligned_labels.append(PAD_LABEL_ID)
            previous_word_id = word_id
        aligned.append(aligned_labels)
    return aligned


def tokenize_and_align_examples(
    examples: Mapping[str, Sequence[Any]], tokenizer: Any, *, max_length: int = 128
) -> dict[str, Any]:
    """Tokenize a CoNLL batch and attach aligned labels for Trainer."""
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
    )
    tokenized["labels"] = align_token_labels(tokenized, examples["ner_tags"])
    return tokenized


def load_conll2003(dataset_script: str | Path, cache_dir: str | Path | None = None) -> Any:
    """Load the provided CoNLL-2003 builder, including Datasets 5 compatibility.

    Datasets 5 no longer executes Python dataset builder scripts through
    ``load_dataset``. When that restriction applies, this function uses the
    supplied script's URL, split names, and parser to create a native
    ``DatasetDict``. The data source and parsing logic therefore remain the
    supplied ``conll2003.py`` rather than a different dataset implementation.
    """
    from datasets import load_dataset

    cache_path = None if cache_dir is None else str(cache_dir)
    try:
        return load_dataset(str(dataset_script), cache_dir=cache_path)
    except RuntimeError as error:
        if "Dataset scripts are no longer supported" not in str(error):
            raise
        return _load_legacy_conll_builder(Path(dataset_script), cache_path)


def _load_legacy_conll_builder(dataset_script: Path, cache_dir: str | None) -> Any:
    """Run a trusted local legacy builder's parser with Datasets 5 primitives."""
    if not dataset_script.is_file():
        raise FileNotFoundError(f"CoNLL builder script not found: {dataset_script}")
    spec = importlib.util.spec_from_file_location("entitylens_conll2003", dataset_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import CoNLL builder script: {dataset_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from datasets import (
        ClassLabel,
        Dataset,
        DatasetDict,
        DownloadConfig,
        Features,
        Value,
    )
    from datasets import (
        Sequence as DatasetSequence,
    )
    from datasets.download.download_manager import DownloadManager

    download_config = DownloadConfig(cache_dir=cache_dir)
    manager = DownloadManager(dataset_name="conll2003", download_config=download_config)
    extracted_path = Path(manager.download_and_extract(module._URL))
    features = Features(
        {
            "id": Value("string"),
            "tokens": DatasetSequence(Value("string")),
            "pos_tags": DatasetSequence(Value("string")),
            "chunk_tags": DatasetSequence(Value("string")),
            "ner_tags": DatasetSequence(ClassLabel(names=list(LABELS))),
        }
    )
    files = {
        "train": extracted_path / module._TRAINING_FILE,
        "validation": extracted_path / module._DEV_FILE,
        "test": extracted_path / module._TEST_FILE,
    }
    return DatasetDict(
        {
            split: Dataset.from_list(
                [
                    example
                    for _, example in module.Conll2003._generate_examples(None, str(filepath))
                ],
                features=features,
            )
            for split, filepath in files.items()
        }
    )


def dataset_diagnostics(
    dataset: Mapping[str, Any], label_names: Sequence[str] = LABELS
) -> DatasetDiagnostics:
    """Compute label, entity, sentence-length, and IOB2-quality summaries."""
    label_counts: Counter[str] = Counter()
    entity_counts: Counter[str] = Counter()
    lengths: list[int] = []
    invalid_count = 0
    split_sizes: dict[str, int] = {}
    for split_name, split in dataset.items():
        split_sizes[str(split_name)] = len(split)
        for row in split:
            tokens = row["tokens"]
            tags = [label_names[tag] if isinstance(tag, int) else tag for tag in row["ner_tags"]]
            lengths.append(len(tokens))
            label_counts.update(tags)
            entity_counts.update(tag[2:] for tag in tags if tag.startswith("B-"))
            invalid_count += len(find_iob2_issues(tags))
    length_summary = {
        "min": float(min(lengths)) if lengths else 0.0,
        "max": float(max(lengths)) if lengths else 0.0,
        "mean": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
    }
    return DatasetDiagnostics(
        split_sizes=split_sizes,
        entity_frequencies=dict(sorted(entity_counts.items())),
        label_frequencies=dict(sorted(label_counts.items())),
        sentence_length=length_summary,
        invalid_iob2_tags=invalid_count,
    )


def validated_tags(tags: Sequence[str], *, repair: bool = False) -> list[str]:
    """Validate a sequence, optionally repairing invalid IOB2 starts."""
    issues = find_iob2_issues(tags)
    if issues and not repair:
        locations = ", ".join(str(issue.index) for issue in issues[:5])
        raise ValueError(f"Invalid IOB2 sequence at positions: {locations}")
    return normalize_iob2(tags) if repair else list(tags)
