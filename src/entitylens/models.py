"""Masked LSTM-family token classifiers used in the EntityLens comparison."""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as functional
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF

ArchitectureName = Literal["lstm", "bilstm", "bilstm_crf"]


class LSTMTokenClassifier(nn.Module):
    """An LSTM or BiLSTM token classifier with packed masked sequences."""

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        *,
        embedding_dim: int = 100,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        pad_id: int = 0,
        bidirectional: bool = False,
        embedding_weights: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        if embedding_weights is not None:
            if embedding_weights.shape != self.embedding.weight.shape:
                raise ValueError(
                    "Pretrained embedding weights do not match the configured vocabulary"
                )
            with torch.no_grad():
                self.embedding.weight.copy_(embedding_weights)
                self.embedding.weight[pad_id].zero_()
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
        )
        encoded_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoded_dim, num_labels)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Return token logits, while keeping padded positions out of the LSTM state."""
        lengths = attention_mask.sum(dim=1).cpu()
        embedded = self.dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.encoder(packed)
        encoded, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=input_ids.size(1)
        )
        return self.classifier(self.dropout(encoded))

    def loss(self, input_ids: Tensor, labels: Tensor, attention_mask: Tensor) -> Tensor:
        """Calculate masked token cross-entropy using -100 as the ignored label."""
        logits = self(input_ids, attention_mask)
        return functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)

    @torch.inference_mode()
    def decode(self, input_ids: Tensor, attention_mask: Tensor) -> list[list[int]]:
        """Greedily decode non-padding positions only."""
        predictions = self(input_ids, attention_mask).argmax(dim=-1)
        return [
            sequence[mask.bool()].tolist()
            for sequence, mask in zip(predictions, attention_mask, strict=True)
        ]


class BiLSTMCRFTokenClassifier(nn.Module):
    """A BiLSTM whose emissions are globally decoded by a masked CRF."""

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        *,
        embedding_dim: int = 100,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        pad_id: int = 0,
        embedding_weights: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.encoder = LSTMTokenClassifier(
            vocab_size,
            num_labels,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pad_id=pad_id,
            bidirectional=True,
            embedding_weights=embedding_weights,
        )
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Return BiLSTM emissions before structured decoding."""
        return self.encoder(input_ids, attention_mask)

    def loss(self, input_ids: Tensor, labels: Tensor, attention_mask: Tensor) -> Tensor:
        """Return negative log likelihood with pads removed from CRF loss."""
        emissions = self(input_ids, attention_mask)
        mask = attention_mask.bool()
        safe_labels = labels.masked_fill(~mask, 0)
        return -self.crf(emissions, safe_labels, mask=mask, reduction="mean")

    @torch.inference_mode()
    def decode(self, input_ids: Tensor, attention_mask: Tensor) -> list[list[int]]:
        """Decode the highest-scoring legal label sequence per sentence."""
        return self.crf.decode(self(input_ids, attention_mask), mask=attention_mask.bool())


def create_classical_model(
    architecture: ArchitectureName,
    vocab_size: int,
    num_labels: int,
    **kwargs: Any,
) -> LSTMTokenClassifier | BiLSTMCRFTokenClassifier:
    """Instantiate one of the comparable classical architectures."""
    if architecture == "lstm":
        return LSTMTokenClassifier(vocab_size, num_labels, bidirectional=False, **kwargs)
    if architecture == "bilstm":
        return LSTMTokenClassifier(vocab_size, num_labels, bidirectional=True, **kwargs)
    if architecture == "bilstm_crf":
        return BiLSTMCRFTokenClassifier(vocab_size, num_labels, **kwargs)
    raise ValueError(f"Unsupported classical architecture: {architecture}")
