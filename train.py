"""
train.py — Fine-tune YOLOv8 on the CODEBRIM bridge defect dataset.

Usage:
    python train.py

Prereqs:
    1. Download CODEBRIM from https://zenodo.org/record/2579133
    2. Convert to YOLO format using prepare_codebrim.py
    3. Ensure data/codebrim.yaml exists
"""

from ultralytics import YOLO
import os

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL  = "yolov8n.pt"        # Start from pretrained nano model
DATA_YAML   = "data/codebrim.yaml"
EPOCHS      = 50
IMG_SIZE    = 640
BATCH       = 16
PROJECT_DIR = "runs/train"
RUN_NAME    = "bridge_defect_v1"
DEVICE      = "0"                  # "0" for GPU, "cpu" for CPU


def main():
    if not os.path.exists(DATA_YAML):
        print(f"ERROR: {DATA_YAML} not found.")
        print("Run prepare_codebrim.py first to convert the dataset.")
        return

    print(f"Loading base model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    print(f"Starting training for {EPOCHS} epochs on {DATA_YAML}")
    results = model.train(
        data    = DATA_YAML,
        epochs  = EPOCHS,
        imgsz   = IMG_SIZE,
        batch   = BATCH,
        project = PROJECT_DIR,
        name    = RUN_NAME,
        device  = DEVICE,
        patience= 10,          # Early stopping
        save    = True,
        plots   = True,
    )

    # Copy best weights to models/
    best_weights = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")
    if os.path.exists(best_weights):
        os.makedirs("models", exist_ok=True)
        import shutil
        shutil.copy(best_weights, "models/bridge_defect_yolov8.pt")
        print("Best weights saved to models/bridge_defect_yolov8.pt")

    print("Training complete.")
    print(f"Results saved to {PROJECT_DIR}/{RUN_NAME}")


if __name__ == "__main__":
    main()
