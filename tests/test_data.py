from entitylens.data import PAD_LABEL_ID, align_token_labels, build_vocabulary, pad_sequences


class FakeBatchEncoding:
    def word_ids(self, batch_index: int):
        assert batch_index == 0
        return [None, 0, 1, 1, None]


def test_vocabulary_and_padding_are_deterministic() -> None:
    vocabulary = build_vocabulary([["Paris", "works"], ["Paris"]])
    padded, masks = pad_sequences([[2, 3], [4]], pad_value=0)

    assert vocabulary.encode(["paris", "missing"]) == [2, 1]
    assert padded == [[2, 3], [4, 0]]
    assert masks == [[1, 1], [1, 0]]


def test_subword_label_alignment_ignores_special_and_continuation_tokens() -> None:
    labels = align_token_labels(FakeBatchEncoding(), [[1, 3]])

    assert labels == [[PAD_LABEL_ID, 1, 3, PAD_LABEL_ID, PAD_LABEL_ID]]
