from pathlib import Path

from entitylens.data import dataset_diagnostics


def test_dataset_diagnostics_counts_labels_entities_and_lengths() -> None:
    dataset = {
        "train": [
            {"tokens": ["Ada", "works"], "ner_tags": [1, 0]},
            {"tokens": ["Paris"], "ner_tags": [5]},
        ]
    }

    summary = dataset_diagnostics(dataset)

    assert summary.split_sizes == {"train": 2}
    assert summary.entity_frequencies == {"LOC": 1, "PER": 1}
    assert summary.sentence_length == {"min": 1.0, "max": 2.0, "mean": 1.5}
    assert summary.invalid_iob2_tags == 0


def test_prepare_data_write_json_round_trip(tmp_path: Path) -> None:
    from scripts.prepare_data import write_json

    destination = tmp_path / "nested" / "value.json"
    write_json(destination, {"b": 1, "a": 2})

    assert destination.read_text(encoding="utf-8") == '{\n  "a": 2,\n  "b": 1\n}'
