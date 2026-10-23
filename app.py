"""CUB-200 Bird Species Classifier — Streamlit application.

Runs ResNet18 checkpoints produced by the user's Colab notebooks.
Inference is CPU-compatible and uses the ImageNet evaluation transform.
No dataset is needed to serve predictions.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import json

import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
NUM_CLASSES = 200

# Both notebook variants are supported. Add only the checkpoints you actually have.
CHECKPOINT_CHOICES = {
    "Partial Fine-Tuning · combined experiment": "resnet18_partial_finetune.pth",
    "Full Fine-Tuning · combined experiment": "resnet18_full_finetune.pth",
    "Frozen Backbone · combined experiment": "resnet18_head_only.pth",
    "Partial Fine-Tuning · earlier experiment": "resnet18_final.pth",
}

st.set_page_config(
    page_title="BirdLens | CUB-200 Classifier",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(165deg, #f5f8fa 0%, #edf2f5 100%);}
    .block-container {padding-top: 2rem; max-width: 1220px;}
    [data-testid="stSidebar"] {background: #e7f0ee;}
    .hero {border-radius: 18px; padding: 1.6rem 2rem; background:
           linear-gradient(110deg, #133b46 0%, #155e63 100%); color: #fff;}
    .hero h1 {font-size: 2.4rem; margin: 0; color: #fff;}
    .hero p {font-size: 1.05rem; margin-top: .55rem; color: #dbeef0;}
    .eyebrow {font-size: .8rem; font-weight: 700; text-transform: uppercase;
              letter-spacing: .15em; color: #a8dbe0;}
    .section-title {font-size: 1.3rem; font-weight: 720; color: #153841; margin-top: 0.8rem;}
    .footer-note {font-size: .82rem; color: #64747a; margin-top: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading ResNet18 checkpoint…")
def load_checkpoint(path_str: str):
    """Cache the trained model across reruns; never download pretrained weights."""
    checkpoint = torch.load(path_str, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain 'model_state_dict'.")
    if int(checkpoint.get("num_classes", NUM_CLASSES)) != NUM_CLASSES:
        raise ValueError("Expected 200 bird species in the checkpoint.")

    names = checkpoint.get("class_names")
    if names is None:
        names_path = ROOT / "class_names.json"
        if not names_path.exists():
            raise ValueError("Missing class names: include class_names in the checkpoint or upload class_names.json.")
        names = json.loads(names_path.read_text(encoding="utf-8"))

    if isinstance(names, dict):
        names = [names[str(i)] for i in range(NUM_CLASSES)]
    if not isinstance(names, (list, tuple)) or len(names) != NUM_CLASSES:
        raise ValueError("Checkpoint must supply exactly 200 ordered class names.")
    names = [str(name) for name in names]
    if len(set(names)) != NUM_CLASSES:
        raise ValueError("Species names must be unique to display prediction scores.")

    model = resnet18(weights=None)  # No network access needed at deployment.
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, NUM_CLASSES),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    return model, names, checkpoint


@st.cache_data

def preprocess_information():
    return "Resize to 256 px → centre crop 224×224 → ImageNet normalisation"


EVAL_TRANSFORM = ResNet18_Weights.IMAGENET1K_V1.transforms()


def predict_species(image: Image.Image, model: nn.Module, class_names: list[str]) -> pd.DataFrame:
    """Return the five most likely species with original 200-way softmax scores."""
    tensor = EVAL_TRANSFORM(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        logits = model(tensor)
        scores = torch.softmax(logits, dim=1).squeeze(0)
        top_scores, top_indices = torch.topk(scores, k=5)
    return pd.DataFrame(
        {
            "Rank": list(range(1, 6)),
            "Bird species": [class_names[int(i)] for i in top_indices.tolist()],
            "Score (%)": [round(float(v) * 100, 2) for v in top_scores.tolist()],
        }
    )


def load_csv_or_none(filename: str):
    path = ROOT / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, OSError, UnicodeDecodeError, ValueError):
        return None


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Applied Machine Learning · CUB-200</div>
      <h1>BirdLens</h1>
      <p>Bird species recognition powered by transfer learning with ResNet18.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

available = {
    label: MODEL_DIR / filename
    for label, filename in CHECKPOINT_CHOICES.items()
    if (MODEL_DIR / filename).is_file()
}

with st.sidebar:
    st.header("Prediction settings")
    st.caption("Choose a trained model checkpoint.")
    if available:
        selected_label = st.selectbox("Training method", list(available.keys()))
        selected_path = available[selected_label]
        st.success("Checkpoint available")
    else:
        selected_label = None
        selected_path = None
        st.warning("No trained model has been added yet.")
        st.code("models/resnet18_final.pth")
    st.divider()
    st.markdown("**Dataset**")
    st.write("200 bird species · CUB-200")
    st.markdown("**Model architecture**")
    st.write("ImageNet-pretrained ResNet18 with a 200-class head")
    st.markdown("**Inference**")
    st.write("CPU · local processing · no training required")

predict_tab, results_tab, about_tab = st.tabs(
    ["Identify a bird", "Model comparison", "About the project"]
)

with predict_tab:
    st.markdown('<div class="section-title">Upload a bird photograph</div>', unsafe_allow_html=True)
    st.write("Upload a clear photo of a bird. The application shows the top five species predictions.")

    left, right = st.columns([1, 1.08], gap="large")
    with left:
        upload = st.file_uploader(
            "Choose a JPG, PNG, or WebP image",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
        )
        image = None
        if upload is not None:
            try:
                image = Image.open(BytesIO(upload.getvalue())).convert("RGB")
                st.image(image, caption="Your uploaded photograph", use_container_width=True)
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("This file could not be opened as an image. Please upload another image.")
        predict_clicked = st.button(
            "Identify bird species",
            type="primary",
            use_container_width=True,
            disabled=(image is None or selected_path is None),
        )

    with right:
        st.markdown('<div class="section-title">Your prediction</div>', unsafe_allow_html=True)
        if predict_clicked and image is not None and selected_path is not None:
            try:
                model, names, metadata = load_checkpoint(str(selected_path))
                predictions = predict_species(image, model, names)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                st.error(f"Unable to load the model or predict this image: {exc}")
            else:
                top_species = str(predictions.iloc[0]["Bird species"])
                top_score = float(predictions.iloc[0]["Score (%)"])
                st.subheader(top_species)
                st.metric("Top prediction score", f"{top_score:.2f}%")
                st.caption(f"Model used: {selected_label}")
                if top_score < 35:
                    st.info("The leading score is relatively low. Review the other top-five species too.")

                st.markdown("**Five most likely species**")
                st.bar_chart(
                    predictions.set_index("Bird species")["Score (%)"],
                    horizontal=True,
                    x_label="Softmax score (%)",
                )
                st.dataframe(
                    predictions.style.format({"Score (%)": "{:.2f}"}),
                    hide_index=True,
                    use_container_width=True,
                )
                st.download_button(
                    "Download prediction table (CSV)",
                    data=predictions.to_csv(index=False).encode("utf-8"),
                    file_name="birdlens_prediction.csv",
                    mime="text/csv",
                )
        else:
            st.info("Your top-five predictions will appear here after you upload an image and press Identify.")
        st.caption("Scores are 200-class softmax outputs, not calibrated confidence probabilities. A photo outside the 200 species may still receive a prediction.")

with results_tab:
    st.markdown('<div class="section-title">Model performance</div>', unsafe_allow_html=True)
    st.write("Comparison of frozen-backbone, partial fine-tuning, and full fine-tuning ResNet18.")

    final_df = load_csv_or_none("FINAL_RESULTS_TABLE.csv")
    if final_df is not None and {
        "training_method", "top1_accuracy", "average_class_accuracy"
    }.issubset(final_df.columns):
        st.subheader("Final test results")
        cols = [
            c for c in [
                "training_method", "epochs", "top1_accuracy",
                "average_class_accuracy", "top5_accuracy",
                "correct_predictions", "test_images"
            ] if c in final_df.columns
        ]
        st.dataframe(final_df[cols].round(2), hide_index=True, use_container_width=True)
        chart_df = final_df.set_index("training_method")[["top1_accuracy", "average_class_accuracy"]]
        chart_df.columns = ["Top-1 Accuracy (%)", "Average Class Accuracy (%)"]
        st.bar_chart(chart_df, horizontal=True)
        st.caption("Test results are descriptive; do not use test accuracy to choose or tune a model.")
    else:
        st.info("To show the final test comparison, upload `FINAL_RESULTS_TABLE.csv` from your combined training notebook into the application folder.")

    cv_df = load_csv_or_none("cv_summary.csv")
    if cv_df is not None and {
        "training_method", "mean_top1", "mean_macro"
    }.issubset(cv_df.columns):
        st.subheader("Three-fold cross-validation")
        show_cols = [
            c for c in ["training_method", "mean_top1", "mean_macro", "sd_macro", "median_best_epoch"]
            if c in cv_df.columns
        ]
        st.dataframe(cv_df[show_cols].round(2), hide_index=True, use_container_width=True)
        st.caption("Training-only cross-validation was used for model selection.")
    else:
        st.info("Optionally upload `cv_summary.csv` to display validation results.")

with about_tab:
    st.markdown('<div class="section-title">How BirdLens works</div>', unsafe_allow_html=True)
    st.markdown(
        """
        **Task:** Fine-grained image classification of 200 Caltech-UCSD Birds species.

        **Architecture:** ResNet18, initialized with ImageNet pretrained weights, with
        the classification head replaced by dropout and a 200-class linear layer.

        **Training strategies:**
        - **Frozen:** Train only the classification head.
        - **Partial fine-tuning:** Update the last residual block and head.
        - **Full fine-tuning:** Update all learnable model parameters.

        **Selection protocol:** Stratified 3-fold cross-validation on the cleaned
        training data. Identical-image leakage and conflicting training labels were
        audited before training. Final test results are reported separately.

        **Loss and optimizer:** Cross-entropy loss and AdamW.
        """
    )
    st.write("**Inference preprocessing:**", preprocess_information())
    st.warning(
        "Research demonstration, not a definitive biological identification tool. "
        "The model only knows 200 classes and may be inaccurate for unfamiliar species, "
        "unusual lighting, or obstructed birds."
    )
    st.write("**For your assignment:** demonstrate image upload, five predictions, and cross-validation/test-result tables in your video.")

st.markdown(
    '<p class="footer-note">BirdLens · COS30082 Applied Machine Learning · CUB-200 · ResNet18</p>',
    unsafe_allow_html=True,
)
