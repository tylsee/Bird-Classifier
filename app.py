from __future__ import annotations

import csv
import html
import io
import json
from datetime import date
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18


# ---------------------------------------------------------------------------
# APP CONFIGURATION
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
NUM_CLASSES = 200
MAX_FILE_MB = 10

# Each model can use checkpoints from the combined notebook OR the earlier run.
MODEL_FILES = {
    "Partial Fine-Tuning ResNet18": [
        "resnet18_partial_finetune.pth",
        "resnet18_final.pth",
    ],
    "Frozen ResNet18": ["resnet18_head_only.pth"],
    "Full Fine-Tuning ResNet18": ["resnet18_full_finetune.pth"],
}

MODEL_TO_MODE = {
    "Frozen ResNet18": "head_only",
    "Partial Fine-Tuning ResNet18": "partial_finetune",
    "Full Fine-Tuning ResNet18": "full_finetune",
}

st.set_page_config(
    page_title="Bird Classifier AI | Bird Identification",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": "Bird Classifier AI — COS30082 Applied Machine Learning project"},
)

# No server-side copy of uploaded files is deliberately retained.
Image.MAX_IMAGE_PIXELS = 30_000_000

STYLES = """
<style>
:root {
    --forest: #205f4f;
    --forest-dark: #174b3e;
    --sage: #ebf3ee;
    --paper: #f7faf8;
    --body: #54665d;
    --line: #dde9e2;
}
.stApp, [data-testid="stAppViewContainer"] {
    background: var(--paper);
    color: #193e34;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { opacity: 0.4; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.block-container {
    max-width: 1215px;
    padding-top: 1.15rem;
    padding-bottom: 1.6rem;
}
@media (max-width: 740px) {
  .block-container { padding: 0.8rem 1rem 1.8rem; }
  .hero-title { font-size: 2.35rem !important; }
  .hero { padding: 2rem 0 1.5rem !important; }
}
* { scroll-behavior: smooth; }
h1, h2, h3, p { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.logo-bar {
    display: flex; align-items: center; gap: 12px; height: 47px;
    font-size: 1.22rem; font-weight: 750; color: var(--forest);
    letter-spacing: -0.7px;
}
.logo-mark {
    width: 37px; height: 37px; background: var(--forest); border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
}
.logo-mark svg { width: 22px; height: 22px; stroke: white; fill: none; stroke-width: 1.8; }
.nav-shadow { border-top: 1px solid #e9efea; margin-top: 0.5rem; }
.hero { text-align: center; padding: 2.5rem 0 1.9rem; }
.eyebrow {
    display: inline-flex; border: 1px solid var(--line); border-radius: 50px;
    padding: 8px 15px; background: white; color: #578575;
    letter-spacing: 2px; text-transform: uppercase; font: 650 0.67rem monospace;
}
.hero-title {
    font-weight: 760;
    color: var(--forest);
    font-size: clamp(2.5rem, 4.5vw, 3.65rem);
    letter-spacing: -2.2px;
    line-height: 1.1;
    margin: 17px auto 12px;
    max-width: 760px;
    text-align: center;
    display: block;
}
.hero-subtitle { color: #62736a; font-size: 1.08rem; margin: 0; }
.panel-heading { color: var(--forest); font-size: 1.07rem; font-weight: 700; margin: 0 0 0.7rem; }
[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"] {
    border-color: var(--line);
}
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 22px !important;
    border-color: var(--line) !important;
    background: white;
}
/* Text-input/drop zone matching the visual references */
[data-testid="stFileUploader"] {
    border: none;
}
[data-testid="stFileUploader"] section {
    min-height: 215px;
    border: 1.6px dashed #c7decf !important;
    background: #fcfefd;
    border-radius: 15px;
}
[data-testid="stFileUploader"] section:hover { border-color: #78af95 !important; }
[data-testid="stFileUploader"] section > div { padding: 1.3rem !important; }
[data-testid="stFileUploader"] button {
    border: 1px solid #cbded1 !important;
    color: var(--forest) !important;
    background-color: white !important;
}
.stButton button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background: var(--forest) !important;
    border: none !important;
    border-radius: 10px !important;
    min-height: 48px;
    color: #fff !important;
    font-weight: 700;
}
.stButton button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
    background: var(--forest-dark) !important;
}
.stButton button:disabled { opacity: 0.42; }
.stButton button[kind="secondary"], button[data-testid="stBaseButton-secondary"] {
    border: none;
    border-radius: 22px;
    background: #edf4ef;
    color: var(--forest);
    font-weight: 600;
}
[data-testid="stSelectbox"] label p { color: var(--forest) !important; font-weight: 650; }
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    border: 1px solid var(--line); border-radius: 10px; background: white;
}
.help-text { color: #84928b; font-size: 0.76rem; text-align: center; margin: 8px 0 0; }
.empty-state {
    min-height: 315px; display: flex; flex-direction: column;
    justify-content: center; align-items: center; gap: 13px;
    color: #6a7b73; text-align: center;
}
.empty-icon {
    height: 61px; width: 61px; border-radius: 50%;
    display: grid; place-items: center; background: #eaf4ee;
    font-size: 1.65rem; color: var(--forest);
}
.empty-state p { max-width: 270px; font-size: 0.95rem; line-height: 1.5; }
.result-top { background: #eff6f1; border: 1px solid #d9ebe0; border-radius: 15px; padding: 16px; }
.result-kicker { font-size: 0.72rem; letter-spacing: 1.3px; text-transform: uppercase;
    color: #678c77; font-weight: 700; }
.result-name { color: var(--forest-dark); font-size: 1.55rem; font-weight: 800;
    letter-spacing: -0.5px; margin: 7px 0; line-height: 1.15; }
.result-score { color: var(--forest); font-size: 1.6rem; font-weight: 750; }
.result-caption { color: #687e71; font-size: 0.79rem; margin-top: 4px; }
.rank-title { color: var(--forest); font-weight: 750; margin: 17px 0 12px; }
.rank-row { margin: 11px 0; }
.rank-meta { display: flex; justify-content: space-between; align-items: center;
    gap: 12px; color: #335f4f; font-size: 0.85rem; margin-bottom: 6px; }
.rank-meta b { font-weight: 750; }
.rank-name { overflow-wrap: anywhere; }
.bar-track { height: 9px; background: #e5eee8; border-radius: 20px; overflow: hidden; }
.bar-fill { height: 100%; background: #66a58a; border-radius: 20px; }
.bar-fill.top { background: var(--forest); }
.section-title { text-align: center; color: var(--forest); font-size: 1.9rem;
    letter-spacing: -0.9px; font-weight: 800; margin: 48px 0 4px; }
.section-subtitle { text-align: center; color: #7b8b82; font-size: 0.9rem; margin: 0 0 24px; }
.metric-card { background: #fff; border: 1px solid var(--line); border-radius: 16px;
    padding: 18px 10px; text-align: center; }
.metric-label { color: #7e8e85; letter-spacing: 1.2px; font: 700 0.68rem monospace; }
.metric-value { font-size: 1.95rem; color: var(--forest); font-weight: 800; margin: 8px 0 1px; }
.table-shell { border: 1px solid var(--line); border-radius: 15px; overflow: hidden;
    margin-top: 22px; background: white; }
.compare-table { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
.compare-table th { background: #f7faf8; padding: 15px 17px; text-align: left;
    color: var(--forest-dark); font-size: 0.79rem; }
.compare-table td { padding: 16px 17px; color: #376b58; border-top: 1px solid #e8eee9; }
.compare-table td:first-child { font-weight: 650; }
.compare-table .pending { color: #a2aaa4; font-style: italic; }
.table-note { color: #8b978f; text-align: center; font-size: 0.77rem; margin-top: 11px; }
.about-wrap { max-width: 850px; margin: 0 auto; }
.about-hero { padding: 3.5rem 0 1.5rem; }
.about-hero h1 { color: var(--forest); font-size: 2.55rem; font-weight: 800; letter-spacing: -1.1px; }
.about-hero p { color: #5d7166; font-size: 1.04rem; margin: 3px 0 0; }
.info-card { border: 1px solid var(--line); border-radius: 19px; background: white;
    padding: 25px 27px; margin-bottom: 17px; }
.info-card h3 { font-size: 1.05rem; text-transform: uppercase; letter-spacing: 1.35px;
    margin: 0 0 15px; color: var(--forest); font-weight: 700; }
.info-card p { margin: 0 0 11px; color: #62746a; line-height: 1.75; font-size: 0.96rem; }
.info-card p:last-child { margin-bottom: 0; }
.info-card.alert { background: #f7f7f1; border-color: #e8dfc3; }
.steps { display: grid; gap: 13px; }
.step { display: flex; align-items: center; gap: 13px; color: #62746a; font-size: 0.96rem; }
.step-num { background: #eaf3ed; color: var(--forest); width: 25px; height: 25px;
    border-radius: 50%; display: grid; place-items: center; font-size: 0.77rem; font-weight: 750; }
.site-footer { border-top: 1px solid #e5eee8; margin-top: 64px;
    padding: 25px 0 3px; display: flex; align-items: center;
    justify-content: space-between; gap: 15px; font-size: 0.80rem; color: #839189; }
.site-footer b { color: var(--forest); }
@media (max-width: 560px) {
 .compare-table { font-size: 0.72rem; }
 .compare-table th, .compare-table td { padding: 10px 8px; }
 .site-footer { flex-direction: column; align-items: flex-start; }
 .about-hero h1 { font-size: 2rem; }
}
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# DATA AND INFERENCE
# ---------------------------------------------------------------------------
EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def discover_models() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for label, candidate_names in MODEL_FILES.items():
        for filename in candidate_names:
            path = MODEL_DIR / filename
            if path.is_file():
                found[label] = path
                break
    return found


def read_class_names(checkpoint: dict) -> list[str]:
    names = checkpoint.get("class_names")
    if names is None:
        for file in [BASE_DIR / "class_names.json", MODEL_DIR / "class_names.json"]:
            if file.is_file():
                names = json.loads(file.read_text(encoding="utf-8"))
                break
    if isinstance(names, dict):
        names = [names.get(str(i), names.get(i)) for i in range(NUM_CLASSES)]
    if not isinstance(names, (list, tuple)) or len(names) != NUM_CLASSES:
        raise ValueError("Checkpoint must contain 200 ordered class names or provide class_names.json.")
    names = [str(name).strip() for name in names]
    if any(not name or name == "None" for name in names):
        raise ValueError("Invalid class names in checkpoint.")
    if len(set(names)) != NUM_CLASSES:
        raise ValueError("Bird class names must be unique.")
    return names


@st.cache_resource(show_spinner="Loading trained bird classifier…")
def load_model(path: str, modified_timestamp_ns: int):
    """Load only YOUR trained checkpoint, never download an ImageNet model."""
    _ = modified_timestamp_ns  # Bust cache if model file changes.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not include 'model_state_dict'.")
    if int(checkpoint.get("num_classes", NUM_CLASSES)) != NUM_CLASSES:
        raise ValueError("Expected a checkpoint with 200 bird classes.")

    names = read_class_names(checkpoint)
    model = resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, NUM_CLASSES),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, names


def predict_bird(upload_bytes: bytes, model_path: Path) -> list[dict]:
    with Image.open(io.BytesIO(upload_bytes)) as img:
        corrected = ImageOps.exif_transpose(img)
        rgb = corrected.convert("RGB")
        tensor = EVAL_TRANSFORM(rgb).unsqueeze(0)

    model, class_names = load_model(str(model_path), model_path.stat().st_mtime_ns)
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = logits.softmax(dim=1)[0]
        scores, indices = probabilities.topk(5)

    return [
        {"species": class_names[int(index)], "score": float(score) * 100}
        for score, index in zip(scores.cpu(), indices.cpu())
    ]


def find_result_file(filename: str) -> Path | None:
    for candidate in [BASE_DIR / filename, BASE_DIR / "results" / filename]:
        if candidate.is_file():
            return candidate
    return None


def read_csv_rows(filename: str) -> list[dict]:
    p = find_result_file(filename)
    if p is None:
        return []
    try:
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except (OSError, UnicodeError, csv.Error):
        return []


def metric_text(value) -> str:
    try:
        return f"{float(value):.2f}%"
    except (ValueError, TypeError):
        return '<span class="pending">Pending</span>'


def normalize_method(text: str) -> str | None:
    key = text.strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"head_only", "frozen_resnet18", "frozen_backbone", "frozen"}:
        return "Frozen ResNet18"
    if key in {"partial_finetune", "partial_fine_tuning", "partial_fine_tuning_resnet18"}:
        return "Partial Fine-Tuning"
    if key in {"full_finetune", "full_fine_tuning", "full_fine_tuning_resnet18"}:
        return "Full Fine-Tuning"
    return None


def model_performance():
    # Historical results are shown until newer matching experiment files are added.
    cv = {
        "Frozen ResNet18": (53.89, 53.35),
        "Partial Fine-Tuning": (65.22, 64.71),
        "Full Fine-Tuning": (None, None),
    }
    for row in read_csv_rows("cv_summary.csv"):
        label = normalize_method(row.get("mode", "") or row.get("training_method", ""))
        if label and row.get("mean_top1") and row.get("mean_macro"):
            try:
                cv[label] = (float(row["mean_top1"]), float(row["mean_macro"]))
            except ValueError:
                pass

    # This represents the earlier partial fine-tuning FINAL TEST evaluation.
    stats = {"top1": 69.27, "macro": 69.06, "top5": 90.37, "images": 1204}
    stats_note = "Previously evaluated Partial Fine-Tuning ResNet18 · 1,204 test images."
    for row in read_csv_rows("FINAL_RESULTS_TABLE.csv") or read_csv_rows("final_test_results_all_methods.csv"):
        label = normalize_method(row.get("mode", "") or row.get("training_method", ""))
        if label == "Partial Fine-Tuning":
            try:
                stats = {
                    "top1": float(row["top1_accuracy"]),
                    "macro": float(row["average_class_accuracy"]),
                    "top5": float(row["top5_accuracy"]),
                    "images": int(float(row["test_images"])),
                }
                stats_note = f"Partial Fine-Tuning ResNet18 · {stats['images']:,} test images (combined run)."
            except (ValueError, KeyError, TypeError):
                pass
    return cv, stats, stats_note


# ---------------------------------------------------------------------------
# SHARED HEADER AND FOOTER
# ---------------------------------------------------------------------------
BIRD_SVG = """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
<path d="M16 7h.01M11 20l1-5M6 21l2-5M20 5c-2.7 2.3-5.5 3.1-8.2 3C7 7.8 4 10.9 4 15c0 1.9.6 3.5 2 4 2.4 1.2 5.9-.3 7-3 5.5.1 8-3.4 7-11Z"/>
</svg>"""

if "page" not in st.session_state:
    st.session_state.page = "Home"

nav_logo, nav_spacer, nav_home, nav_about = st.columns([4.9, 3.0, 1.0, 1.45], vertical_alignment="center")
with nav_logo:
    st.markdown(
        f'<div class="logo-bar"><span class="logo-mark">{BIRD_SVG}</span>Bird Classifier AI</div>',
        unsafe_allow_html=True,
    )
with nav_home:
    if st.button("Home", key="nav_home", use_container_width=True, type="secondary"):
        st.session_state.page = "Home"
        st.rerun()

st.markdown('<div class="nav-shadow"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------------
if st.session_state.page == "Home":
    st.markdown(
        """
        <div class="hero">
          <span class="eyebrow">CUB-200 · 200 Bird Species</span>
          <h1 class="hero-title">Identify What Bird You're<br>Looking At</h1>
          <p class="hero-subtitle">Upload a bird photo and let our AI identify its species.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    available = discover_models()
    if not available:
        st.warning(
            "Bird identification is not connected yet. Add a trained ResNet18 "
            "checkpoint to the models/ folder to activate predictions."
        )
    
    upload_panel, prediction_panel = st.columns([1, 1], gap="medium")

    with upload_panel:
        with st.container(border=True):
            st.markdown('<div class="panel-heading">Upload Bird Photo</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Drop your bird photo here, or click to browse",
                type=["jpg", "jpeg", "png", "webp"],
                label_visibility="visible",
                accept_multiple_files=False,
                key="bird_upload",
            )
            st.markdown('<div class="help-text">JPG · JPEG · PNG · WEBP</div>', unsafe_allow_html=True)

            image_bytes = None
            if uploaded is not None:
                image_bytes = uploaded.getvalue()
                if len(image_bytes) > MAX_FILE_MB * 1024 * 1024:
                    st.error(f"Please upload an image smaller than {MAX_FILE_MB} MB.")
                    image_bytes = None
                else:
                    try:
                        with Image.open(io.BytesIO(image_bytes)) as im:
                            im.verify()
                        with Image.open(io.BytesIO(image_bytes)) as im:
                            preview = ImageOps.exif_transpose(im)
                            st.image(preview.copy(), use_container_width=True)
                    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                        st.error("This file is not a valid image. Please choose another photograph.")
                        image_bytes = None

            if available:
                models = list(available)
                selected_name = st.selectbox("AI Model", options=models, index=0)
                selected_path = available[selected_name]
            else:
                selected_name = "Partial Fine-Tuning ResNet18"
                selected_path = None
                st.selectbox("AI Model", ["No checkpoint installed"], disabled=True)

            uploaded_signature = None
            if image_bytes is not None:
                import hashlib
                uploaded_signature = hashlib.sha256(image_bytes).hexdigest()
            signature = (uploaded_signature, selected_name)
            if st.session_state.get("prediction_signature") != signature:
                st.session_state.pop("predictions", None)

            predict_clicked = st.button(
                "Identify Bird",
                type="primary",
                use_container_width=True,
                disabled=(image_bytes is None or selected_path is None),
                key="identify_bird",
            )
            if predict_clicked and selected_path is not None and image_bytes is not None:
                with st.spinner("Analysing bird features…"):
                    try:
                        predictions = predict_bird(image_bytes, selected_path)
                    except Exception as exc:
                        st.error(f"Prediction failed. Check the model checkpoint. Details: {exc}")
                    else:
                        st.session_state.predictions = predictions
                        st.session_state.prediction_signature = signature

            st.markdown(
                '<div class="help-text">Images are processed in memory and not saved by this app.</div>',
                unsafe_allow_html=True,
            )

    with prediction_panel:
        with st.container(border=True):
            st.markdown('<div class="panel-heading">Prediction Results</div>', unsafe_allow_html=True)
            predictions = st.session_state.get("predictions")
            if predictions and st.session_state.get("prediction_signature") == signature:
                top = predictions[0]
                top_species = html.escape(top["species"])
                st.markdown(
                    f"""
                    <div class="result-top">
                      <div class="result-kicker">Top Prediction</div>
                      <div class="result-name">{top_species}</div>
                      <div class="result-score">{top['score']:.1f}%</div>
                      <div class="result-caption">Model: {html.escape(selected_name)}</div>
                    </div>
                    <div class="rank-title">Top 5 Predictions</div>
                    """,
                    unsafe_allow_html=True,
                )
                bars = []
                for rank, item in enumerate(predictions, start=1):
                    width = min(max(item["score"], 0), 100)
                    bars.append(
                        f'<div class="rank-row">'
                        f'<div class="rank-meta"><span class="rank-name">{rank}. {html.escape(item["species"])}</span>'
                        f'<b>{item["score"]:.1f}%</b></div>'
                        f'<div class="bar-track"><div class="bar-fill {"top" if rank == 1 else ""}" style="width:{width:.2f}%"></div></div>'
                        f'</div>'
                    )
                st.markdown("".join(bars), unsafe_allow_html=True)
                st.caption("Scores are softmax outputs, not guaranteed calibrated probabilities.")
            else:
                st.markdown(
                    """
                    <div class="empty-state">
                       <span class="empty-icon">◎</span>
                       <p>Your bird identification results will appear here.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    cv_stats, test_stats, stats_note = model_performance()
    st.markdown('<h2 class="section-title">AI Model Performance</h2>', unsafe_allow_html=True)
    st.markdown(f'<p class="section-subtitle">{html.escape(stats_note)}</p>', unsafe_allow_html=True)

    stat_cols = st.columns(3, gap="medium")
    for col, label, value in zip(
        stat_cols,
        ["TOP-1 ACCURACY", "AVERAGE CLASS ACCURACY", "TOP-5 ACCURACY"],
        [test_stats["top1"], test_stats["macro"], test_stats["top5"]],
    ):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{float(value):.2f}%</div></div>',
                unsafe_allow_html=True,
            )

    table_rows = ""
    for method in ["Frozen ResNet18", "Partial Fine-Tuning", "Full Fine-Tuning"]:
        acc, macro = cv_stats[method]
        table_rows += (
            f"<tr><td>{html.escape(method)}</td>"
            f"<td>{metric_text(acc)}</td><td>{metric_text(macro)}</td></tr>"
        )
    st.markdown(
        f"""
        <div class="table-shell">
            <table class="compare-table">
                <thead><tr>
                    <th>Training Method</th>
                    <th>Top-1 Accuracy (Cross-Validation)</th>
                    <th>Average Class Accuracy (Cross-Validation)</th>
                </tr></thead>
                <tbody>{table_rows}</tbody>
            </table>
        </div>
        <p class="table-note">Cross-validation results are distinct from final test accuracies.
        Missing values are shown as pending, never estimated.</p>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f'<div class="site-footer"><b>Bird Classifier AI | Applied Machine Learning Project</b>'
    f'<span>© {date.today().year} Bird Classifier AI</span></div>',
    unsafe_allow_html=True,
)
