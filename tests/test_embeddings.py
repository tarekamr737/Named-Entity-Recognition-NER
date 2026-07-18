import torch

from entitylens.data import Vocabulary
from entitylens.embeddings import load_glove_embeddings
from entitylens.models import LSTMTokenClassifier


def test_glove_loader_aligns_known_tokens_and_keeps_padding_zero(tmp_path) -> None:
    vectors = tmp_path / "glove.txt"
    vectors.write_text("hello 0.1 0.2\nworld 0.3 0.4\n", encoding="utf-8")
    vocabulary = Vocabulary({"<pad>": 0, "<unk>": 1, "hello": 2, "missing": 3})

    report = load_glove_embeddings(vectors, vocabulary)
    model = LSTMTokenClassifier(
        vocab_size=4,
        num_labels=9,
        embedding_dim=2,
        embedding_weights=report.weights,
    )

    assert report.matched_tokens == 1
    assert report.coverage == 0.5
    assert torch.allclose(model.embedding.weight[2], torch.tensor([0.1, 0.2]))
    assert torch.equal(model.embedding.weight[0], torch.zeros(2))
