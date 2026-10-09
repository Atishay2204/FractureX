import io
import os
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
from ultralytics import YOLO

app = FastAPI(title="FractureX API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VERIFIED_CLASSES = (
    "elbow_fracture",
    "finger_fracture",
    "forearm_fracture",
    "humerus_fracture",
    "shoulder_fracture",
    "wrist_fracture",
)
VERIFIED_CLASS_SCHEMAS = (VERIFIED_CLASSES, ("fracture",))

# Load model
candidates = [
    os.environ.get("FRACTURE_MODEL_PATH"),
    "best_binary_fracture.pt",
    "best_verified_light.pt",
]
model_path = next((path for path in candidates if path and os.path.isfile(path)), None)
if model_path is None:
    raise FileNotFoundError(
        "Corrected model weights are missing. Add best_verified_light.pt to the "
        "deployment or set FRACTURE_MODEL_PATH to the verified checkpoint."
    )
model = YOLO(model_path)

def display_name(name: str) -> str:
    return str(name).replace("_", " ").strip().capitalize()

CLASS_NAMES = {
    int(class_id): display_name(name)
    for class_id, name in model.names.items()
}

MODEL_CLASSES = tuple(str(model.names[i]) for i in sorted(model.names))
if MODEL_CLASSES not in VERIFIED_CLASS_SCHEMAS:
    raise RuntimeError("Unsupported checkpoint")

IS_BINARY_FRACTURE_MODEL = MODEL_CLASSES == ("fracture",)

def to_base64_png(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.post("/predict")
async def predict(file: UploadFile = File(...), confidence: int = Form(5)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        # Match the corrected model's 640px training resolution.
        results = model.predict(
            np.array(image),
            imgsz=640,
            conf=confidence / 100,
            iou=0.45,
            max_det=3,
            verbose=False,
        )
        
        annotated = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
        
        rows = [
            {
                "Region": CLASS_NAMES.get(int(b.cls[0]), f"Class {int(b.cls[0])}"),
                "Confidence": float(b.conf[0]) * 100,
            }
            for b in results[0].boxes
        ]
        rows.sort(key=lambda r: r["Confidence"], reverse=True)
        
        annotated_b64 = to_base64_png(annotated)
        
        return JSONResponse({
            "rows": rows,
            "annotated_image": annotated_b64,
            "is_binary": IS_BINARY_FRACTURE_MODEL
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
