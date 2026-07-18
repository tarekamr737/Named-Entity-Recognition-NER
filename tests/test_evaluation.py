from entitylens.evaluation import (
    boundary_error_summary,
    evaluate_sequences,
    label_sequences_from_ids,
)


def test_entity_metrics_are_strict_and_reported_per_type() -> None:
    true, predicted = label_sequences_from_ids([[1, 2, 0]], [[1, 2, -100]])
    result = evaluate_sequences(true, predicted)

    assert result.f1 == 1.0
    assert result.per_entity["PER"].support == 1
    assert result.per_entity["ORG"].support == 0


def test_boundary_summary_distinguishes_exact_overlap_and_spurious_spans() -> None:
    summary = boundary_error_summary(
        ["B-PER", "I-PER", "O", "B-LOC", "O"],
        ["B-PER", "O", "B-ORG", "B-LOC", "O"],
    )

    assert summary.exact_matches == 1
    assert summary.boundary_errors == 1
    assert summary.missed_entities == 0
    assert summary.spurious_entities == 1
    assert summary.complete_entity_accuracy == 0.5
