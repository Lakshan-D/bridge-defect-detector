"""
prepare_codebrim.py — Convert CODEBRIM dataset to YOLO format.

CODEBRIM structure:
    CODEBRIM/
        crack/
        spalling/
        corrosion/
        efflorescence/
        exposed_rebar/
        background/

This script:
    1. Reads images from each class folder
    2. Creates YOLO-format labels (whole-image bounding boxes as baseline)
    3. Splits into train/val/test (70/20/10)
    4. Writes data/codebrim.yaml

Usage:
    python prepare_codebrim.py --src /path/to/CODEBRIM
"""

import os
import argparse
import shutil
import random
from pathlib import Path

CLASSES = ["crack", "spalling", "corrosion", "efflorescence", "exposed_rebar", "background"]

SPLITS = {"train": 0.70, "val": 0.20, "test": 0.10}


def write_yolo_label(label_path: Path, class_id: int):
    """Write a whole-image bounding box label (x_c y_c w h all = 0.5 1.0)."""
    with open(label_path, "w") as f:
        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")


def main(src: str, dst: str = "data/codebrim"):
    src_path = Path(src)
    dst_path = Path(dst)

    for split in SPLITS:
        (dst_path / split / "images").mkdir(parents=True, exist_ok=True)
        (dst_path / split / "labels").mkdir(parents=True, exist_ok=True)

    all_files = []
    for cls_id, cls_name in enumerate(CLASSES):
        cls_dir = src_path / cls_name
        if not cls_dir.exists():
            print(f"WARNING: {cls_dir} not found, skipping.")
            continue
        imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        for img in imgs:
            all_files.append((img, cls_id))

    random.seed(42)
    random.shuffle(all_files)

    n = len(all_files)
    n_train = int(n * SPLITS["train"])
    n_val   = int(n * SPLITS["val"])

    split_map = (
        [(f, c, "train") for f, c in all_files[:n_train]] +
        [(f, c, "val")   for f, c in all_files[n_train:n_train + n_val]] +
        [(f, c, "test")  for f, c in all_files[n_train + n_val:]]
    )

    for img_path, cls_id, split in split_map:
        dst_img = dst_path / split / "images" / img_path.name
        dst_lbl = dst_path / split / "labels" / (img_path.stem + ".txt")
        shutil.copy(img_path, dst_img)
        write_yolo_label(dst_lbl, cls_id)

    # Write YAML
    yaml_content = f"""# CODEBRIM — Concrete Bridge Defect Dataset
# https://zenodo.org/record/2579133

path: {dst_path.resolve()}
train: train/images
val:   val/images
test:  test/images

nc: {len(CLASSES)}
names: {CLASSES}
"""
    yaml_path = Path("data/codebrim.yaml")
    yaml_path.parent.mkdir(exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Dataset prepared: {n} images across {len(CLASSES)} classes.")
    print(f"Train: {n_train} | Val: {n_val} | Test: {n - n_train - n_val}")
    print(f"YAML written to {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Path to downloaded CODEBRIM folder")
    parser.add_argument("--dst", default="data/codebrim", help="Output path")
    args = parser.parse_args()
    main(args.src, args.dst)
