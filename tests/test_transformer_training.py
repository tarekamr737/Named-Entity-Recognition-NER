import numpy as np

from entitylens.transformer_training import transformer_metrics


def test_transformer_metrics_ignores_special_tokens() -> None:
    logits = np.array([[[0.0, 4.0], [4.0, 0.0], [0.0, 4.0]]])
    labels = np.array([[1, 0, -100]])

    metrics = transformer_metrics((logits, labels))

    assert metrics["f1"] == 1.0
    assert metrics["per_f1"] == 1.0
