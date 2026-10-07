# Bone Fracture Detection

A Streamlit-based X-ray fracture screening demo. The deployed application uses a binary YOLO detector with one class: `fracture`.

> This project is for education and demonstration only. It is not a medical diagnosis tool; X-rays must be reviewed by a qualified clinician.

## Run locally

```powershell
py -3.12 -m pip install -r requirements.txt
streamlit run app.py
```

The deployed checkpoint is `best_binary_fracture.pt` in the repository root. To temporarily use another compatible checkpoint, set `FRACTURE_MODEL_PATH` to its path.

## Repository layout

```text
app.py                         Streamlit application
best_binary_fracture.pt        Active binary fracture checkpoint
best_verified_light.pt         Legacy six-class fallback checkpoint
notebooks/                     Kaggle training and evaluation notebooks
src/                           Dataset preparation and training utilities
outputs/                       Downloaded models, reports, and visualizations
tools/legacy/                  One-off migration scripts retained for history
```

## Current model

The active model was fine-tuned using BoneFractureYolo8, FracAtlas, and GRAZPEDWRI-DX. Its held-out test results were:

- Precision: 88.8%
- Recall: 78.8%
- mAP@50: 87.3%
- mAP@50–95: 48.5%

Training selections are recorded in `outputs/reports/model_selection.txt`.
