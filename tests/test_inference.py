import torch

from entitylens.data import Vocabulary
from entitylens.inference import (
    PredictedEntity,
    _entities_from_iob,
    _token_spans,
    predict_classical,
    rule_fallback,
)


class FixedTokenClassifier:
    """A deterministic stand-in for exercising the inference adapter."""

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((*input_ids.shape, 9))
        logits[0, 0, 1] = 6.0  # B-PER
        logits[0, 1, 2] = 6.0  # I-PER
        return logits

    def decode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[list[int]]:
        return [[1, 2]]


def test_iob_spans_repair_an_illegal_start_and_preserve_offsets() -> None:
    text = "Ada Lovelace arrived."
    tokens = _token_spans(text)
    entities = _entities_from_iob(text, tokens, ["I-PER", "I-PER", "O", "O"], [0.9] * 4)

    assert entities == [
        PredictedEntity("Ada Lovelace", "PER", 0.9, 0, 12),
    ]


def test_classical_inference_is_deterministic() -> None:
    vocabulary = Vocabulary({"<pad>": 0, "<unk>": 1, "ada": 2, "lovelace": 3})
    model = FixedTokenClassifier()

    first = predict_classical("Ada Lovelace", model, vocabulary)
    second = predict_classical("Ada Lovelace", model, vocabulary)

    assert first.entities == second.entities
    assert first.token_tags == second.token_tags
    assert first.entities[0].text == "Ada Lovelace"
    assert first.entities[0].entity_type == "PER"


def test_rule_fallback_is_deterministic_and_local() -> None:
    first = rule_fallback("OpenAI met Microsoft in Cairo.")
    second = rule_fallback("OpenAI met Microsoft in Cairo.")

    assert first.source == second.source == "fallback"
    assert first.entities == second.entities
    assert [(item.text, item.entity_type) for item in first.entities] == [
        ("OpenAI", "ORG"),
        ("Microsoft", "ORG"),
        ("Cairo", "LOC"),
    ]
