# EntityLens

<img width="1920" height="1080" alt="Screenshot 2026-07-19 165650" src="https://github.com/user-attachments/assets/355bf4fe-6b45-43e9-bb79-d39cb2e1bbcc" />

<img width="1920" height="1080" alt="Screenshot 2026-07-19 165710" src="https://github.com/user-attachments/assets/b70d8708-252a-44e8-a461-9aa0beac2a20" />


EntityLens is a reproducible Named Entity Recognition (NER) workbench for the
CoNLL-2003 dataset. It provides a common data and evaluation pipeline for
comparing sequence models with a transformer baseline, then exposes their
predictions through an interactive Streamlit application.

> **Project status:** educational/research workbench. The saved models were
> trained on the full official training split; their reported scores are from
> the validation split, not the held-out test split.

## Highlights

- CoNLL-2003 loading and dataset diagnostics.
- IOB2 validation, normalization, and entity-level evaluation.
- LSTM, BiLSTM, and BiLSTM + CRF token classifiers.
- DistilBERT-style transformer token classification with subword-label alignment.
- Optional GloVe initialization for classical models.
- Reproducible seeds, train-only vocabularies, saved checkpoints, and focused tests.
- Streamlit workbench for token-level predictions, confidence scores, and benchmark review.

## Model benchmark

The following snapshot was generated using the full CoNLL-2003 training split
(14,041 sentences) and the standard validation split (3,250 sentences). The
classical models were trained for eight epochs; the transformer was fine-tuned
for three epochs. The test split was not used to train or select a model.

| Model | Validation F1 | Precision | Recall | Runtime | Parameters | Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LSTM | 56.16% | 61.34% | 51.78% | 313.41 s | 2,470,005 | 9.42 MB |
| BiLSTM | 69.00% | 72.78% | 65.60% | 232.99 s | 2,838,901 | 10.83 MB |
| BiLSTM + CRF | 70.26% | 76.80% | 64.74% | 264.84 s | 2,839,000 | 10.83 MB |
| Transformer | **93.24%** | **92.79%** | **93.70%** | 2,303.48 s | 65,197,833 | 248.72 MB |

The CRF layer improves the recurrent baseline, while the transformer has the
highest validation F1 at a substantially higher compute and storage cost. Run a
final evaluation on the held-out test split before using a model in production.

The source files are generated under `artifacts/metrics/`:

- `model_comparison.json` — benchmark configuration and machine-readable results.
- `model_comparison.csv` — tabular results for further analysis.
- `boundary_analysis.json` — BiLSTM versus CRF entity-boundary diagnostics.

## Dataset

The prepared CoNLL-2003 snapshot contains 14,041 training sentences, 3,250
validation sentences, and 3,453 test sentences. It includes the standard PER,
ORG, LOC, and MISC entity categories. The preparation pipeline reports zero
invalid IOB2 tags in the current dataset diagnostics.

## Installation

The project requires Python 3.10–3.14. Create or activate a virtual environment,
then install the package and development dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

PyTorch wheels depend on the Python version and hardware platform. If pip cannot
select a compatible wheel, use the official PyTorch installation selector and
then install the remaining project dependencies.

## Verify the installation

```powershell
python -m pytest
python -m ruff check .
```

## Reproduce the workflow

### Prepare the data

```powershell
python scripts\prepare_data.py
```

This downloads the dataset through `conll2003.py`, validates the labels, and
creates train-only vocabularies and diagnostics under `artifacts/`.

### Train classical models

Run a full-split experiment:

```powershell
python scripts\train_classical.py lstm
python scripts\train_classical.py bilstm
python scripts\train_classical.py bilstm_crf
```

Supported architectures are `lstm`, `bilstm`, and `bilstm_crf`. Use
`--max-train` and `--max-validation` only for a compact smoke run. Checkpoints
and training metadata are saved under
`artifacts/checkpoints/`.

### Optional GloVe initialization

```powershell
python scripts\download_glove.py --dimension 50
python scripts\train_classical.py bilstm --epochs 1 --max-train 64 --max-validation 32 --glove-file data\embeddings\glove.6B.50d.txt --checkpoint-name bilstm_glove
```

### Fine-tune the transformer

```powershell
python scripts\train_transformer.py --epochs 3
```

Use `--max-train` and `--max-validation` for a compact smoke run. The pipeline
uses dynamic subword-label alignment and the same entity-level metrics.

### Generate the comparison

```powershell
python scripts\export_comparison.py --train-samples 14041 --validation-samples 3250 --note "Full-split validation benchmark; the held-out test split was not used for training."
python scripts\analyze_boundaries.py --validation-samples 3250
```

### Launch the workbench

```powershell
python -m streamlit run app.py
```

Open the local URL shown by Streamlit, normally `http://localhost:8501`. Select a
trained architecture in the sidebar, enter text, and choose **Analyze entities**.
The application reads model artifacts locally and does not write submitted text
to disk.

## Repository layout

```text
app.py                 Streamlit application
conll2003.py           CoNLL-2003 dataset builder
scripts/               Data preparation, training, and evaluation commands
src/entitylens/        Reusable NER package code
tests/                 Unit tests
artifacts/             Local checkpoints and generated reports (ignored)
data/                  Local downloaded/processed data (ignored)
```

## Version-control policy

Downloaded datasets, model checkpoints, embeddings, caches, and generated
reports are excluded from version control. Recreate them with the commands above
after cloning the repository.

## License

EntityLens is released under the [MIT License](LICENSE).
