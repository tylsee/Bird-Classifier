---
title: BirdLens CUB-200 Classifier
emoji: 🐦
colorFrom: teal
colorTo: blue
sdk: docker
app_port: 8501
pinned: false
short_description: Classify 200 bird species with ResNet18 transfer learning
---

# BirdLens — CUB-200 Bird Species Classifier

A Streamlit application for bird species classification using ResNet18 checkpoints.

**Setup:** Add at least one checkpoint to the `models/` folder; see `DEPLOYMENT_GUIDE.md`.

- Upload a bird image to see the five most likely species.
- Select between trained Frozen, Partial, and Full Fine-Tuning checkpoints if included.
- Add exported CSVs to show cross-validation and final test comparisons.

Inference runs on CPU, with no access to Google Drive, Colab, or the original dataset.
