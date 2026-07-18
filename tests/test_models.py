import torch

from entitylens.models import BiLSTMCRFTokenClassifier, LSTMTokenClassifier


def test_lstm_masks_padding_in_loss_and_decode() -> None:
    model = LSTMTokenClassifier(vocab_size=12, num_labels=9, embedding_dim=8, hidden_dim=10)
    input_ids = torch.tensor([[2, 3, 0], [4, 0, 0]])
    masks = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    labels = torch.tensor([[1, 0, -100], [3, -100, -100]])

    loss = model.loss(input_ids, labels, masks)
    decoded = model.decode(input_ids, masks)

    assert torch.isfinite(loss)
    assert [len(sequence) for sequence in decoded] == [2, 1]


def test_crf_decoding_and_loss_respect_masks() -> None:
    model = BiLSTMCRFTokenClassifier(vocab_size=12, num_labels=9, embedding_dim=8, hidden_dim=10)
    input_ids = torch.tensor([[2, 3, 0], [4, 0, 0]])
    masks = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    labels = torch.tensor([[1, 0, -100], [3, -100, -100]])

    loss = model.loss(input_ids, labels, masks)
    decoded = model.decode(input_ids, masks)

    assert torch.isfinite(loss)
    assert [len(sequence) for sequence in decoded] == [2, 1]
