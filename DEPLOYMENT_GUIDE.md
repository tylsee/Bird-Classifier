# Deploy BirdLens to Hugging Face Spaces (Streamlit with Docker)

## 1. Get model checkpoints from Colab

You need at least **one** `.pth` file containing the learned ResNet18 weights.

From the **earlier partial-fine-tuning notebook**:

`/content/drive/MyDrive/CUB200_Assignment1/resnet18_results/resnet18_final.pth`

From the **combined notebook**:

- `resnet18_combined_results/resnet18_head_only.pth`
- `resnet18_combined_results/resnet18_partial_finetune.pth`
- `resnet18_combined_results/resnet18_full_finetune.pth`

To download a checkpoint in Google Colab:

```python
from google.colab import files
files.download('/content/drive/MyDrive/CUB200_Assignment1/resnet18_results/resnet18_final.pth')
```

**Important:** The `.pth` file is essential. `app.py` is only inference code, not a trained model.

Copy the `.pth` file(s) to `models/` **without changing the specified names**.

Checkpoints from these notebooks already store `class_names`. If your checkpoint does not,
include the matching `class_names.json` at the root of this project.

## 2. Optional: Add results tables

From `resnet18_combined_results/`, copy these into the project root:

- `FINAL_RESULTS_TABLE.csv` → shows final test accuracies for each method.
- `cv_summary.csv` → shows the training-only 3-fold validation comparison.

These files are optional. If absent, the app still predicts birds but the comparison
page displays instructions instead of statistics. Do not edit numerical results.

## 3. Test locally (optional)

Install Python 3.11 and run:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local address printed by Streamlit, usually `http://localhost:8501`.

## 4. Host on Hugging Face Spaces (recommended for the assignment)

1. Create or sign in to a Hugging Face account: https://huggingface.co/
2. Go to https://huggingface.co/new-space
3. Choose a Space name, for example `birdlens-cub200`, and select **Docker** as the SDK.
4. Choose public visibility if your lecturer must access the app without signing in.
5. Upload the project files **to the Space repository root**, including `README.md`,
   `Dockerfile`, `requirements.txt`, `app.py`, `.streamlit/config.toml`,
   `models/<your-checkpoint>.pth`, and optional comparison CSVs.
6. The Space's README already specifies `sdk: docker` and `app_port: 8501`.
7. Let the Space build from the Dockerfile. Check the **Logs** if there are errors.
8. Open the **App** tab and test an actual bird image.
9. Share the Space link (for example `https://huggingface.co/spaces/YOUR_USERNAME/birdlens-cub200`)
   in the assignment PDF and video.

The ZIP itself should **not** be uploaded as the only Space file. Extract it first.
Files larger than ordinary Git limits may use Git LFS; `.gitattributes` is included.

## 5. Alternative: Streamlit Community Cloud

If your lecturer accepts a `streamlit.app` URL instead of a Hugging Face Space,
put the same files and the model(s) into a GitHub repository and create an app at
https://share.streamlit.io/ (entry point `app.py`). However, the assignment PDF
calls for a **Hugging Face link**, so the Docker Space is the recommended host.

## 6. How to demonstrate in your video

- Open the live hosted site.
- Explain the selected training strategy and architecture.
- Upload a bird photo, run inference, and interpret Top-5 scores.
- Open the model comparison tab and show CV vs test metrics.
- Explain limitations: 200-class coverage, similar species, imperfect calibration.

## 7. Troubleshooting

- **No trained model has been added yet:** put a checkpoint into `models/` using one of the supported filenames.
- **Mismatched `fc` layer or state dict:** your checkpoint may come from a different architecture; use the ResNet18 checkpoint trained by the supplied notebooks.
- **Model appears frozen/stuck:** first load can take a few seconds on CPU; subsequent runs reuse a cached model.
- **Wrong bird names:** check the `class_names` inside the checkpoint; the app uses the checkpoint order, not filename sorting.
- **Result table missing:** upload the CSV exports into the project root.
- **App build failed:** inspect the Hugging Face Space build logs for dependency or memory problems.

## 8. Grading caution

The rubric text explicitly mentions **Gradio**, and the assignment PDF requests a
**Hugging Face** app link. Deploying Streamlit inside a Hugging Face Docker Space
meets the hosting requirement but is not literally a Gradio app. Use this framework
for submission only if the lecturer permits Streamlit.
