# TASKS.md

## 1. Project Setup
- [x] Create project structure and virtual environment.
- [x] Install PyTorch, Transformers, Datasets, seqeval, TorchCRF, Streamlit, and visualization dependencies.
- [x] Configure Impeccable Skill for UI implementation and Chrome DevTools MCP for browser testing.
- [x] Add configuration, logging, reproducibility seeds, and model artifact folders.

## 2. Data Pipeline
- [x] Load CoNLL-2003 using conll2003.py.
- [x] Explore label distribution, sentence lengths, and entity frequencies.
- [x] Validate and map labels to the IOB scheme.
- [x] Build word, character/subword, and label vocabularies.
- [x] Load GloVe or FastText embeddings.
- [x] Handle OOV words with character-level embeddings or subword tokenization.
- [x] Pad sequences and create attention masks.
- [x] Align transformer subword tokens with entity labels.

## 3. Model Development
- [x] Implement LSTM token classifier.
- [x] Implement BiLSTM token classifier.
- [x] Implement BiLSTM + CRF with masked CRF loss and decoding.
- [x] Fine-tune BERT or DistilBERT using `AutoTokenizer`, `AutoModelForTokenClassification`, and `Trainer`.
- [x] Save checkpoints, vocabularies, label maps, tokenizer, and training metadata.

## 4. Training and Evaluation
- [x] Train all four models using consistent data splits.
- [x] Track loss, validation F1, runtime, and model size.
- [x] Evaluate with seqeval precision, recall, and F1.
- [x] Report PER, ORG, LOC, and MISC results separately.
- [x] Compare complete-entity accuracy and boundary errors.
- [x] Analyze where CRF improves valid transitions and span boundaries.
- [x] Select the best deployment model.

## 5. Visualization and Error Analysis
- [x] Display token-level predictions and confidence scores.
- [x] Highlight predicted spans using entity-specific colors.
- [x] Add confusion and boundary-error summaries.
- [x] Include representative correct and incorrect predictions.

## 6. Streamlit UI and Deployment
- [x] Use Impeccable Skill to implement the approved `PRODUCTS.md` UI direction.
- [x] Build a polished single-page Streamlit inference application.
- [x] Add free-text input and example sentences.
- [x] Highlight PER, ORG, LOC, and MISC spans in real time.
- [x] Show entity table with text, type, confidence, and position.
- [x] Add model selector for architecture comparison.
- [x] Cache model loading and validate empty or oversized input.
- [x] Add model limitations and privacy notice.
- [x] Check responsive layout, accessibility, loading, empty, success, and error states.
- [x] Document local launch instructions.

## 7. Browser Testing with Chrome DevTools MCP
- [x] Launch the Streamlit app and inspect it through Chrome DevTools MCP.
- [x] Test text submission, model switching, entity highlighting, tables, thresholds, and clear/reset flows.
- [x] Verify desktop and mobile viewport layouts.
- [x] Inspect console errors, failed network requests, accessibility issues, and rendering problems.
- [x] Measure page load and inference interaction performance.
- [x] Capture screenshots for all key states and fix discovered issues.

## 8. Quality and Delivery
- [x] Add unit tests for preprocessing, label alignment, CRF decoding, and inference.
- [x] Verify deterministic inference and graceful error handling.
- [x] Add README with setup, training, evaluation, and deployment instructions.
- [x] Export final metrics, comparison table.
- [x] Run the APP and gimme it running live.
