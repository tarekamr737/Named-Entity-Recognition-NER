# PRODUCTS.md

## Product
**Name:** EntityLens  
**Type:** Interactive Named Entity Recognition analysis tool  
**Primary users:** NLP students, researchers, analysts, and developers validating text extraction models.

## Product Goal
Allow users to enter free text, detect named entities instantly, inspect confidence and span boundaries, and compare predictions across LSTM, BiLSTM, BiLSTM + CRF, and Transformer models.

## UI Direction
A focused **developer tool / NLP workbench** rather than a marketing dashboard.

**Personality:** Clear, technical, trustworthy  
**Avoid:** Excessive gradients, decorative animations, crowded dashboards, and oversized cards.

## Page Structure

### 1. Header
- Product name and one-line description.
- Compact model-status indicator.
- Optional link to methodology or repository.

### 2. Input Workspace
- Large multiline text area.
- Example-text selector.
- Primary `Analyze Entities` button.
- Secondary `Clear` action.
- Character or token count.

### 3. Entity Visualization
- Render the original text with color-coded entity spans.
- Use consistent accessible colors:
  - PER — blue
  - ORG — purple
  - LOC — green
  - MISC — orange
- Include a visible legend.
- Preserve whitespace and punctuation.
- Show a neutral empty state before analysis.

### 4. Entity Results
Display a compact table containing:
- Entity text
- Entity type
- Confidence
- Start and end positions

Support sorting by confidence or entity type.

### 5. Model Comparison
- Model selector in the sidebar or a compact segmented control.
- Optional comparison mode for viewing two models.
- Show:
  - Architecture
  - Validation F1
  - Inference time
  - Parameter count
- Clearly mark the recommended deployment model.

### 6. Explainability and Diagnostics
Use an expandable section for:
- Token-level labels
- Confidence scores
- IOB tags
- Boundary disagreements
- Invalid transitions corrected by CRF

### 7. Footer
- Dataset and model version.
- Limitations notice.
- Privacy statement indicating that entered text is processed only for inference.

## Streamlit Layout
- Use `st.set_page_config(layout="wide")`.
- Main content: text input and highlighted output.
- Sidebar: model selection, confidence threshold, and display options.
- Use two columns beneath the input:
  - Left: highlighted text
  - Right: entity table
- Place comparison metrics and diagnostics below the primary result.
- Cache models with `st.cache_resource`.

## Interaction Rules
- Analyze only when the user clicks the primary button.
- Disable analysis for empty input.
- Display a spinner during inference.
- Show helpful messages when no entities are found.
- Retain the submitted text while users change visualization options.
- Never expose raw stack traces to users.

## Accessibility
- Target WCAG AA contrast.
- Do not rely on color alone; show entity labels inside or beside spans.
- Support keyboard navigation.
- Use readable typography and a minimum 16 px body size.
- Avoid motion unless essential.
- Ensure highlighted spans remain understandable in dark and light themes.

## Responsive Behavior
- Desktop: two-column results layout.
- Tablet and mobile: stack highlighted output above the entity table.
- Keep the primary action visible without excessive scrolling.

## Core Components
- `AppHeader`
- `TextInputPanel`
- `ModelSelector`
- `EntityLegend`
- `HighlightedText`
- `EntityTable`
- `ModelMetrics`
- `TokenDiagnostics`
- `EmptyState`
- `ErrorMessage`

## MVP Acceptance Criteria
- Users can submit free text and receive highlighted entity spans.
- PER, ORG, LOC, and MISC labels are visually distinguishable.
- Entity positions and confidence scores are displayed.
- Users can switch between available trained models.
- The application handles empty input and model errors safely.
- The interface works in both light and dark modes.
