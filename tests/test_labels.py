from entitylens.labels import find_iob2_issues, is_legal_transition, normalize_iob2


def test_illegal_i_tag_is_reported_and_repaired() -> None:
    tags = ["O", "I-ORG", "I-ORG", "I-PER"]

    issues = find_iob2_issues(tags)

    assert [issue.index for issue in issues] == [1, 3]
    assert normalize_iob2(tags) == ["O", "B-ORG", "I-ORG", "B-PER"]


def test_iob2_transition_rules() -> None:
    assert is_legal_transition("B-LOC", "I-LOC")
    assert not is_legal_transition("B-LOC", "I-ORG")
    assert not is_legal_transition("O", "I-PER")
