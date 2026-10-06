"""Train and validate the corrected six-class fracture detector.

Run this after ``merge_datasets.py``.  The merger deliberately excludes the
unverified FracAtlas anatomy labels, so all labels in the resulting YAML have
a known body region.

Example (Kaggle):
    python src/train_verified_model.py --data /kaggle/working/merged_data.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


VERIFIED_CLASSES = [
    "elbow_fracture",
    "finger_fracture",
    "forearm_fracture",
    "humerus_fracture",
    "shoulder_fracture",
    "wrist_fracture",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Merged YOLO data YAML")
    parser.add_argument("--model", default="yolov8l.pt", help="Base YOLO checkpoint")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=0)
    parser.add_argument("--project", type=Path, default=Path("runs"))
    parser.add_argument("--name", default="fracture_verified_v2")
    return parser.parse_args()


def validate_dataset(data_path: Path) -> None:
    with data_path.open(encoding="utf-8") as data_file:
        config = yaml.safe_load(data_file)
    names = config.get("names", [])
    if isinstance(names, dict):
        names = [names[index] for index in sorted(names)]
    if names != VERIFIED_CLASSES:
        raise ValueError(
            "Refusing to train with unverified classes. Expected "
            f"{VERIFIED_CLASSES}, got {names}."
        )


def main() -> None:
    args = parse_args()
    validate_dataset(args.data)

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        patience=50,
        seed=42,
        plots=True,
        close_mosaic=30,
    )

    best_path = Path(results.save_dir) / "weights" / "best.pt"
    best_model = YOLO(best_path)
    model_classes = [best_model.names[index] for index in sorted(best_model.names)]
    if model_classes != VERIFIED_CLASSES:
        raise RuntimeError(f"Saved checkpoint has the wrong class schema: {model_classes}")

    metrics = best_model.val(data=str(args.data), split="test", imgsz=args.imgsz, device=args.device)
    print(f"Verified checkpoint: {best_path}")
    print(f"mAP@50: {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    for name, ap in zip(VERIFIED_CLASSES, metrics.box.ap50):
        print(f"{name}: {ap:.4f}")


if __name__ == "__main__":
    main()
