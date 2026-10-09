# Bone Fracture Detection

A Streamlit-based X-ray fracture screening demo. The deployed application uses a binary YOLO detector with one class: `fracture`.

> This project is for education and demonstration only. It is not a medical diagnosis tool; X-rays must be reviewed by a qualified clinician.

## Run locally

```powershell
py -3.12 -m pip install -r requirements.txt
streamlit run app.py
```

## Mobile app prototype

The native client is in `mobile-app/`; its camera and image picker send an image
to the separate FastAPI service in `api.py`. Keep the model weights available to
the API process. Start the API from the repository root:

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 -m uvicorn api:app --host 0.0.0.0 --port 8000
```

In a second terminal, configure the API address in `mobile-app/.env` and start
the Expo client:

```powershell
cd mobile-app
Copy-Item .env.example .env
# Edit .env and set the API URL for your computer or deployment.
npm install
npm start
```

Use `http://10.0.2.2:8000` for an Android emulator, `http://127.0.0.1:8000`
for an iOS simulator, or `http://<computer-LAN-IP>:8000` for a physical phone
on the same Wi-Fi network. For a deployed app, point `EXPO_PUBLIC_API_URL` at
an HTTPS API deployment that has the checkpoint and the Python dependencies.
The Streamlit Cloud website does not automatically host this FastAPI service.

The API reads uploaded images into memory and does not save them. Avoid sending
identifiable patient information to this demonstration model.

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
