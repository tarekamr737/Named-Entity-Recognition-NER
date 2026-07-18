"""EntityLens, a local Streamlit Named Entity Recognition workbench."""
# ruff: noqa: E501

from __future__ import annotations

import html
import json
from typing import Any

import pandas as pd
import streamlit as st

from entitylens.config import Paths
from entitylens.inference import (
    InferenceResult,
    PredictedEntity,
    available_architectures,
    load_classical_model,
    load_transformer_model,
    predict_classical,
    predict_transformer,
    rule_fallback,
)

st.set_page_config(page_title="EntityLens Workbench", page_icon="◈", layout="wide")

EXAMPLES = {
    "Board meeting": (
        "The executive board of Microsoft met in Seattle yesterday. Satya Nadella discussed "
        "European expansion in Zurich with the United Nations."
    ),
    "Research note": "OpenAI researchers presented a new language model in London on Tuesday.",
    "Clear input": "",
}
ENTITY_META = {
    "PER": ("Person", "per"),
    "ORG": ("Organization", "org"),
    "LOC": ("Location", "loc"),
    "MISC": ("Miscellaneous", "misc"),
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { --ink:#e2e9e9; --muted:#91a2a5; --line:#2e4144; --base:#111617; --panel:#182022; --panel-2:#1e292b; --cyan:#62e1e9; --violet:#cbb8ff; }
        .stApp { background: radial-gradient(circle at 94% 2%, rgba(91, 211, 218, .08), transparent 27rem), #111617; color:var(--ink); }
        .stApp, .stApp * { font-family: Aptos, "Segoe UI", sans-serif; }
        [data-testid="stSidebar"] { background:#151d1f; border-right:1px solid var(--line); }
        [data-testid="stSidebar"] > div:first-child { padding-top:1.25rem; }
        [data-testid="stHeader"] { background:transparent; }
        [data-testid="stExpandSidebarButton"] > span,
        [data-testid="stSidebarCollapseButton"] > button > span { display:none !important; }
        [data-testid="stExpandSidebarButton"]::after,
        [data-testid="stSidebarCollapseButton"] > button::after { content:"☰"; color:var(--ink); font:1.25rem/1 Aptos, "Segoe UI", sans-serif; }
        details > summary [data-testid="stIconMaterial"] { display:none !important; }
        details > summary::before { content:"›"; color:var(--cyan); font:1.2rem/1 Aptos, "Segoe UI", sans-serif; margin-right:.5rem; }
        details[open] > summary::before { content:"⌄"; }
        .block-container { max-width: 1440px; padding: 1.2rem 2.4rem 3rem; }
        .topbar { display:flex; align-items:center; justify-content:space-between; padding: .7rem 0 1.2rem; border-bottom:1px solid var(--line); margin-bottom:1.25rem; }
        .brand { color:var(--cyan); font-family:"Cascadia Mono", Consolas, monospace; font-size:1.12rem; font-weight:500; letter-spacing:-.04em; }
        .brand small { color:var(--muted); font-size:.67rem; margin-left:.7rem; letter-spacing:.08em; }
        .system-state { color:#bdeef0; font:500 .65rem "Cascadia Mono", Consolas, monospace; letter-spacing:.08em; border:1px solid #396368; padding:.36rem .52rem; border-radius:4px; }
        .eyebrow { color:var(--muted); font:500 .68rem "Cascadia Mono", Consolas, monospace; text-transform:uppercase; letter-spacing:.1em; }
        .metric { min-height:94px; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:1rem; }
        .metric .value { color:var(--ink); font-size:1.65rem; font-weight:600; line-height:1.25; margin-top:.25rem; }
        .metric .minor { color:var(--muted); font-size:.75rem; margin-top:.22rem; }
        .rail-title { color:var(--cyan); font:500 .7rem "JetBrains Mono", monospace; letter-spacing:.08em; text-transform:uppercase; margin:1.2rem 0 .5rem; }
        .panel-heading { display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--line); padding-bottom:.7rem; margin-bottom:.85rem; }
        .panel-heading strong { font:500 .72rem "JetBrains Mono", monospace; letter-spacing:.09em; text-transform:uppercase; }
        .legend { display:grid; grid-template-columns:1fr 1fr; gap:.65rem; }
        .legend > div { background:#172022; border:1px solid var(--line); border-radius:6px; padding:.72rem; font-size:.75rem; color:var(--muted); }
        .chip, .entity { display:inline-block; font:500 .68rem "Cascadia Mono", Consolas, monospace; border-radius:4px; padding:.12rem .35rem; margin-right:.32rem; }
        .per { color:#a4efff; background:#133d48; border:1px solid #377b8c; }
        .org { color:#decaff; background:#30294b; border:1px solid #695c9a; }
        .loc { color:#b8efcc; background:#173d2d; border:1px solid #397d5b; }
        .misc { color:#ffc89b; background:#4b3020; border:1px solid #9b6747; }
        .result-copy { min-height:180px; background:#141c1e; border:1px solid var(--line); border-radius:7px; padding:1.1rem; color:#d7e1e1; font-size:1rem; line-height:2.25; }
        .result-copy .entity { margin:0 .1rem; padding:.08rem .32rem; white-space:nowrap; }
        .muted-copy { color:var(--muted); padding:1.2rem 0; line-height:1.6; }
        .notice { color:var(--muted); font-size:.75rem; line-height:1.55; border-top:1px solid var(--line); padding-top:1rem; margin-top:1.5rem; }
        .stButton > button { border-radius:4px; min-height:2.45rem; font-weight:600; }
        .stButton > button[kind="primary"] { background:#62e1e9; color:#073033; border-color:#62e1e9; }
        .stTextArea textarea { background:#11191b !important; border:1px solid var(--line) !important; color:var(--ink) !important; border-radius:6px !important; line-height:1.55; }
        .stSelectbox [data-baseweb="select"] > div, .stSlider [data-baseweb="slider"] { background:#182022; }
        .stDataFrame { border:1px solid var(--line); border-radius:6px; overflow:hidden; }
        @media (max-width: 1100px) {
            [data-testid="stHorizontalBlock"] { flex-wrap:wrap; }
            [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex:1 1 100% !important; min-width:0 !important; }
            [data-testid="stHorizontalBlock"]:has(.metric) { flex-wrap:wrap; }
            [data-testid="stHorizontalBlock"]:has(.metric) > [data-testid="stColumn"] { flex:1 1 calc(50% - 1rem) !important; min-width:0 !important; }
        }
        @media (max-width: 900px) { .block-container { padding: .8rem 1rem 2rem; } .topbar { align-items:flex-start; gap:.8rem; } .brand small { display:block; margin:.35rem 0 0; } [data-testid="stHorizontalBlock"]:has(.metric) > [data-testid="stColumn"] { flex:1 1 100% !important; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_backend(architecture: str) -> tuple[Any, ...]:
    checkpoint = Paths().checkpoints / architecture
    if architecture == "transformer":
        return load_transformer_model(checkpoint)
    return load_classical_model(checkpoint, architecture)


def get_metadata(architecture: str) -> dict[str, Any]:
    path = Paths().checkpoints / architecture / "training_metadata.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_metric_artifact(filename: str) -> dict[str, Any] | None:
    """Read optional offline analysis without making the app depend on it."""
    path = Paths().metrics / filename
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_inference(text: str, architecture: str) -> InferenceResult:
    try:
        backend = load_backend(architecture)
        result = predict_transformer(text, *backend) if architecture == "transformer" else predict_classical(text, *backend)
        return result if result.entities else rule_fallback(text)
    except Exception:
        return rule_fallback(text)


def render_highlights(text: str, entities: list[PredictedEntity], threshold: float) -> str:
    selected = [entity for entity in entities if entity.confidence >= threshold]
    if not selected:
        return '<div class="muted-copy">No entity spans meet the current confidence threshold.</div>'
    fragments: list[str] = []
    cursor = 0
    for entity in sorted(selected, key=lambda item: (item.start, item.end)):
        if entity.start < cursor:
            continue
        fragments.append(html.escape(text[cursor:entity.start]))
        label = html.escape(entity.entity_type)
        value = html.escape(text[entity.start:entity.end])
        fragments.append(f'<span class="entity {ENTITY_META[entity.entity_type][1]}">{value} <b>{label}</b></span>')
        cursor = entity.end
    fragments.append(html.escape(text[cursor:]))
    return '<div class="result-copy">' + ''.join(fragments) + '</div>'


def entity_frame(entities: list[PredictedEntity], threshold: float) -> pd.DataFrame:
    rows = [
        {"Entity": item.text, "Type": item.entity_type, "Confidence": f"{item.confidence:.1%}", "Start": item.start, "End": item.end}
        for item in entities
        if item.confidence >= threshold
    ]
    return pd.DataFrame(rows, columns=["Entity", "Type", "Confidence", "Start", "End"])


def clear_input() -> None:
    """Reset the text widget before Streamlit renders it again."""
    st.session_state.input_text = ""
    st.session_state.result = None


inject_styles()
paths = Paths()
architectures = available_architectures(paths.checkpoints)
if not architectures:
    architectures = ["bilstm_crf"]

if "input_text" not in st.session_state:
    st.session_state.input_text = EXAMPLES["Board meeting"]
if "result" not in st.session_state:
    st.session_state.result = None
default_architecture = "bilstm_crf" if "bilstm_crf" in architectures else architectures[0]
if "last_architecture" not in st.session_state:
    st.session_state.last_architecture = default_architecture

with st.sidebar:
    st.markdown('<div class="brand">ENTITYLENS</div>', unsafe_allow_html=True)
    st.markdown('<div class="rail-title">Inference configuration</div>', unsafe_allow_html=True)
    architecture = st.selectbox(
        "Active architecture",
        architectures,
        index=architectures.index(default_architecture),
        format_func=lambda name: name.replace("_", " ").title(),
    )
    threshold = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.10, step=0.05)
    st.checkbox("Show IOB diagnostics", value=False, key="show_diagnostics")
    st.markdown('<div class="rail-title">Workbench</div>', unsafe_allow_html=True)
    st.caption("Architecture\n\nWeights\n\nTokenizer\n\nDictionary\n\nRun log")
    st.markdown('<div class="rail-title">Local processing</div>', unsafe_allow_html=True)
    st.caption("Text stays in this local Python process. No input is sent to a remote service.")

metadata = get_metadata(architecture)
validation = metadata.get("validation", {})
if "eval_f1" in validation:
    f1 = validation["eval_f1"]
    parameter_count = metadata.get("parameter_count", 0)
elif validation:
    f1 = validation.get("f1", 0)
    parameter_count = metadata.get("parameter_count", 0)
else:
    f1, parameter_count = 0, 0

st.markdown(
    '<div class="topbar"><div class="brand">EntityLens.Workbench <small>LOCAL NER ANALYSIS</small></div>'
    f'<div class="system-state">● {architecture.replace("_", " ").upper()} READY</div></div>',
    unsafe_allow_html=True,
)

metrics = st.columns(4, gap="medium")
metric_values = [
    ("Active model", architecture.replace("_", " ").title(), "Local checkpoint"),
    ("Validation F1", f"{f1:.1%}", "Compact training run"),
    ("Parameters", f"{parameter_count / 1_000_000:.1f}M" if parameter_count else "n/a", "Saved artifact metadata"),
    ("Inference", f"{st.session_state.result.elapsed_ms:.0f} ms" if st.session_state.result else "Awaiting run", "Local CPU estimate"),
]
for column, (label, value, note) in zip(metrics, metric_values, strict=True):
    with column:
        st.markdown(f'<div class="metric"><div class="eyebrow">{label}</div><div class="value">{value}</div><div class="minor">{note}</div></div>', unsafe_allow_html=True)

st.write("")
input_column, legend_column = st.columns([7, 5], gap="large")
with input_column:
    st.markdown('<div class="panel-heading"><strong>Text input buffer</strong><span class="eyebrow">Submit when ready</span></div>', unsafe_allow_html=True)
    example = st.selectbox("Load an example", list(EXAMPLES), label_visibility="collapsed")
    if st.button("Load selected example"):
        st.session_state.input_text = EXAMPLES[example]
        st.session_state.result = None
    text = st.text_area("Text for entity extraction", key="input_text", height=220, placeholder="Enter raw text for NER analysis…")
    action_a, action_b, counter = st.columns([1.25, 0.7, 2])
    with action_a:
        analyse = st.button(
            "Analyze entities",
            type="primary",
            width="stretch",
            disabled=not text.strip() or len(text) > 5000,
        )
    with action_b:
        st.button("Clear", width="stretch", on_click=clear_input)
    with counter:
        st.markdown(f'<p class="eyebrow" style="text-align:right; padding-top:.75rem">{len(text):,} characters · {len(text.split()):,} tokens</p>', unsafe_allow_html=True)
    if len(text) > 5000:
        st.error("Keep one request under 5,000 characters so local inference remains responsive.")
    if analyse:
        with st.spinner("Running local inference…"):
            st.session_state.result = run_inference(text, architecture)
            st.session_state.last_architecture = architecture
        entity_count = len(st.session_state.result.entities)
        st.success(f"Analysis complete: {entity_count} candidate {'span' if entity_count == 1 else 'spans'} found.")

with legend_column:
    st.markdown('<div class="panel-heading"><strong>Entity recognition legend</strong></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="legend">'
        '<div><span class="chip per">PER</span><br/>Person names</div>'
        '<div><span class="chip org">ORG</span><br/>Organizations</div>'
        '<div><span class="chip loc">LOC</span><br/>Locations</div>'
        '<div><span class="chip misc">MISC</span><br/>Other named entities</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="rail-title">Confidence threshold</div>', unsafe_allow_html=True)
    st.progress(int(threshold * 100), text=f"Showing spans at {threshold:.0%} confidence or above")
    st.markdown('<div class="rail-title">Output contract</div>', unsafe_allow_html=True)
    st.caption("Every accepted span retains its text, IOB type, confidence, and original character positions.")

st.write("")
st.markdown('<div class="panel-heading"><strong>Analysis result</strong><span class="eyebrow">IOB2 spans · local inference</span></div>', unsafe_allow_html=True)
result = st.session_state.result
left, right = st.columns([1.1, 0.9], gap="large")
with left:
    st.markdown('<div class="eyebrow" style="margin-bottom:.65rem">Interactive visualization</div>', unsafe_allow_html=True)
    if result:
        st.markdown(render_highlights(text, result.entities, threshold), unsafe_allow_html=True)
        if result.source == "fallback":
            st.warning("The selected checkpoint did not produce usable spans for this text. Showing the limited local fallback recognizer instead.")
    else:
        st.markdown('<div class="result-copy"><div class="muted-copy">Run analysis to inspect entity spans in their original context.</div></div>', unsafe_allow_html=True)
with right:
    st.markdown('<div class="eyebrow" style="margin-bottom:.65rem">Extracted entities</div>', unsafe_allow_html=True)
    if result:
        frame = entity_frame(result.entities, threshold)
        if frame.empty:
            st.info("No entities meet the active confidence threshold. Lower it or try another model.")
        else:
            st.dataframe(frame, width="stretch", hide_index=True)
    else:
        st.markdown('<div class="muted-copy">Entity text, type, confidence, and character offsets will appear here.</div>', unsafe_allow_html=True)

if result and st.session_state.show_diagnostics:
    with st.expander("Token diagnostics", expanded=False):
        token_rows = [{"Token": token, "IOB tag": tag, "Confidence": f"{score:.1%}"} for token, tag, score in result.token_tags]
        st.dataframe(pd.DataFrame(token_rows), width="stretch", hide_index=True)

with st.expander("Benchmark and boundary review", expanded=False):
    comparison = get_metric_artifact("model_comparison.json")
    boundaries = get_metric_artifact("boundary_analysis.json")
    if comparison:
        st.caption(
            "Compact 64/32, one-epoch benchmark. Use it to compare workflow behavior, not "
            "deployment quality."
        )
        comparison_rows = pd.DataFrame(comparison["rows"])[
            ["architecture", "validation_f1", "runtime_seconds", "model_size_megabytes"]
        ].rename(
            columns={
                "architecture": "Model",
                "validation_f1": "Validation F1",
                "runtime_seconds": "Runtime (s)",
                "model_size_megabytes": "Size (MB)",
            }
        )
        st.dataframe(comparison_rows, width="stretch", hide_index=True)
        st.info(
            f"Recommended in this compact benchmark: "
            f"{comparison['recommended_architecture'].replace('_', ' ').title()}."
        )
    else:
        st.caption("Run `scripts/export_comparison.py` to add the benchmark comparison here.")

    if boundaries:
        st.markdown('<div class="rail-title">Boundary error summary</div>', unsafe_allow_html=True)
        rows = [
            {
                "Model": model.replace("_", " ").title(),
                "Complete spans": report["summary"]["complete_entity_accuracy"],
                "Boundary errors": report["summary"]["boundary_errors"],
                "Invalid IOB": report["summary"]["invalid_iob2_transitions"],
            }
            for model, report in boundaries["architectures"].items()
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        delta = boundaries["bilstm_to_crf_delta"]
        st.caption(
            "BiLSTM → CRF: "
            f"{delta['complete_entity_accuracy']:+.1%} complete-span accuracy, "
            f"{delta['boundary_errors']:+d} boundary errors, "
            f"{delta['invalid_iob2_transitions']:+d} invalid IOB transitions."
        )
        st.markdown('<div class="rail-title">Top token-label confusions</div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(boundaries["architectures"]["bilstm_crf"]["top_label_confusions"]),
            width="stretch",
            hide_index=True,
        )
        crf_examples = boundaries["architectures"]["bilstm_crf"]["examples"]
        correct_example = crf_examples.get("exact_matches")
        incorrect_example = (
            crf_examples.get("boundary_errors")
            or crf_examples.get("missed_entities")
            or crf_examples.get("spurious_entities")
        )
        if correct_example or incorrect_example:
            st.markdown('<div class="rail-title">Representative validation sentence</div>', unsafe_allow_html=True)
            correct_column, incorrect_column = st.columns(2)
            with correct_column:
                st.caption("Correct span example")
                if correct_example:
                    st.code(" ".join(correct_example["tokens"]), language=None)
                else:
                    st.caption("No exact span was found in this slice.")
            with incorrect_column:
                st.caption("Boundary or missed-span example")
                if incorrect_example:
                    st.code(" ".join(incorrect_example["tokens"]), language=None)
                else:
                    st.caption("No error example was found in this slice.")
            st.caption("Gold and predicted IOB tags are available in `boundary_analysis.json`.")
    else:
        st.caption("Run `scripts/analyze_boundaries.py` to add boundary diagnostics here.")

st.markdown(
    '<div class="notice">CoNLL-2003 labels: PER, ORG, LOC, MISC. The stored checkpoints are compact training artifacts, not production-grade models. '
    'Entered text is processed locally for inference and is not persisted by this application.</div>',
    unsafe_allow_html=True,
)
