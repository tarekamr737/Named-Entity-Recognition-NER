"""Fine-tuning utilities for BERT-family EntityLens token classifiers."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np

from .data import tokenize_and_align_examples
from .evaluation import evaluate_sequences, label_sequences_from_ids
from .labels import ID_TO_LABEL, LABELS


def transformer_metrics(eval_prediction: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """Convert Trainer logits and ignored labels to overall and per-type F1 metrics."""
    logits, labels = eval_prediction
    prediction_ids = np.argmax(logits, axis=-1)
    true_sequences, predicted_sequences = label_sequences_from_ids(prediction_ids, labels)
    evaluation = evaluate_sequences(true_sequences, predicted_sequences)
    metrics = {
        "precision": evaluation.precision,
        "recall": evaluation.recall,
        "f1": evaluation.f1,
    }
    for entity, result in evaluation.per_entity.items():
        metrics[f"{entity.lower()}_f1"] = result.f1
    return metrics


def train_transformer(
    dataset: Any,
    output_dir: str | Path,
    *,
    model_name: str = "distilbert/distilbert-base-cased",
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    epochs: float = 3,
) -> dict[str, Any]:
    """Fine-tune and save a BERT-family token classifier using Hugging Face Trainer."""
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenized = dataset.map(
        lambda examples: tokenize_and_align_examples(examples, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Aligning CoNLL labels to subword tokens",
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id={label: index for index, label in ID_TO_LABEL.items()},
    )
    argument_kwargs: dict[str, Any] = {
        "output_dir": str(destination),
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": epochs,
        "weight_decay": 0.01,
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        # Reloading a full Transformer checkpoint at epoch end doubles peak CPU
        # memory on constrained machines. The final epoch model is already saved
        # and evaluated, so keep the training process single-copy by default.
        "load_best_model_at_end": False,
        "report_to": [],
    }
    parameters = inspect.signature(TrainingArguments).parameters
    evaluation_key = "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
    argument_kwargs[evaluation_key] = "epoch"
    training_arguments = TrainingArguments(**argument_kwargs)
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=transformer_metrics,
    )
    train_result = trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(destination))
    tokenizer.save_pretrained(destination)
    metadata = {
        "architecture": "transformer",
        "base_model": model_name,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training": train_result.metrics,
        "validation": metrics,
    }
    (destination / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=float), encoding="utf-8"
    )
    return metadata
