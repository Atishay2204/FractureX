import base64
import hashlib
import io
import logging
import os
import time
from functools import lru_cache

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


LOGGER = logging.getLogger("fracture_api")
MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_CLASSES = {
    ("fracture",),
    (
        "elbow_fracture",
        "finger_fracture",
        "forearm_fracture",
        "humerus_fracture",
        "shoulder_fracture",
        "wrist_fracture",
    ),
}

app = FastAPI(title="OsteoScan fracture screening API", version="1.0.0")
allowed_origins = [
    origin.strip()
    for origin in os.environ.get(
        "MOBILE_CORS_ORIGINS",
        "http://localhost:8081,http://127.0.0.1:8081",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


@lru_cache(maxsize=1)
def load_model():
    candidates = (
        os.environ.get("FRACTURE_MODEL_PATH"),
        "best_binary_fracture.pt",
        "best_verified_light.pt",
    )
    model_path = next(
        (path for path in candidates if path and os.path.isfile(path)), None
    )
    if model_path is None:
        raise FileNotFoundError(
            "Model weights are missing. Set FRACTURE_MODEL_PATH or add the "
            "verified checkpoint to the API deployment."
        )

    model = YOLO(model_path)
    classes = tuple(str(model.names[index]) for index in sorted(model.names))
    if classes not in ALLOWED_CLASSES:
        raise RuntimeError("The configured model checkpoint is not supported.")
    return model


def display_name(name: str) -> str:
    return str(name).replace("_", " ").strip().capitalize()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(
    image: UploadFile = File(...),
    confidence: int = Form(5, ge=1, le=20),
):
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Upload a JPG or PNG image.")

    image_bytes = image.file.read(MAX_IMAGE_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Images must be 10 MB or smaller.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as uploaded_image:
            original = uploaded_image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The image could not be opened.")

    try:
        model = load_model()
        start = time.perf_counter()
        prediction = model.predict(
            np.asarray(original),
            imgsz=640,
            conf=confidence / 100,
            iou=0.45,
            max_det=3,
            verbose=False,
        )[0]
        elapsed = time.perf_counter() - start
    except FileNotFoundError as error:
        LOGGER.exception("Model checkpoint is unavailable")
        raise HTTPException(status_code=503, detail=str(error))
    except Exception:
        LOGGER.exception("Model inference failed")
        raise HTTPException(status_code=500, detail="Could not analyze this image.")

    annotated = cv2.cvtColor(prediction.plot(), cv2.COLOR_BGR2RGB)
    annotated_buffer = io.BytesIO()
    Image.fromarray(annotated).save(annotated_buffer, format="PNG")
    detections = [
        {
            "region": display_name(model.names[int(box.cls[0])]),
            "confidence": round(float(box.conf[0]) * 100, 1),
        }
        for box in prediction.boxes
    ]
    detections.sort(key=lambda detection: detection["confidence"], reverse=True)

    return {
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "detections": detections,
        "annotated_image": base64.b64encode(annotated_buffer.getvalue()).decode("ascii"),
        "elapsed_seconds": round(elapsed, 3),
        "disclaimer": (
            "Screening demonstration only. This result is not a diagnosis; "
            "consult a qualified clinician."
        ),
    }
