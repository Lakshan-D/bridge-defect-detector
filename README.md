# 🌉 Bridge Defect Detection System

**AI-powered automated defect detection for bridge and civil infrastructure inspection — aligned with the EU-funded [STRUCTURE project](https://structure-project.eu) on UAV-based inspection and digital twin development.**

---

## Demo

> Upload a bridge inspection image or video. The system detects and localises defects in real time.

![Demo placeholder — add a GIF of the app running here](docs/demo.gif)

---

## What It Does

This system uses a fine-tuned **YOLOv8** model to automatically detect and classify concrete defects on bridge surfaces from inspection images or UAV-captured video frames.

**Detects 6 defect classes** from the [CODEBRIM dataset](https://zenodo.org/record/2579133):

| Class | Description |
|---|---|
| Crack | Surface and structural cracks |
| Spalling | Concrete surface breakaway |
| Corrosion | Rebar or surface oxidation |
| Efflorescence | Salt deposit leaching |
| Exposed Rebar | Visible reinforcement steel |
| Background | Non-defect surface |

---

## Why This Matters

Manual bridge inspection is slow, expensive, and dangerous. Automating defect detection from UAV-captured imagery is a core challenge in modern infrastructure maintenance. This project demonstrates the **AI perception layer** of a UAV inspection pipeline — the step between raw image capture and structured digital twin output.

This work directly mirrors the technical approach of the EU STRUCTURE project, which develops AI-enabled UAV inspection and digital twin systems for transportation infrastructure with a focus on bridges.

---

## Tech Stack

- **Model:** YOLOv8 (Ultralytics) — fine-tuned on CODEBRIM
- **App:** Streamlit
- **Vision:** OpenCV, Pillow
- **Training:** PyTorch (GPU via Google Colab)
- **Language:** Python 3.10+

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/lakshan-d/bridge-defect-detector.git
cd bridge-defect-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app (base YOLOv8 weights)
```bash
streamlit run app.py
```

The app runs in your browser at `http://localhost:8501`. On first run it downloads base YOLOv8 weights automatically.

---

## Fine-tune on CODEBRIM (Recommended)

For best results, fine-tune on the real bridge defect dataset.

### Step 1 — Download CODEBRIM
```
https://zenodo.org/record/2579133
```
Extract to a local folder, e.g. `/data/CODEBRIM_raw`.

### Step 2 — Prepare dataset
```bash
python prepare_codebrim.py --src /data/CODEBRIM_raw
```

### Step 3 — Train
```bash
python train.py
```
Fine-tuned weights are saved automatically to `models/bridge_defect_yolov8.pt`.

### Step 4 — Run app with fine-tuned weights
```bash
streamlit run app.py
```
The app detects `models/bridge_defect_yolov8.pt` and loads it automatically.

---

## Project Structure

```
bridge-defect-detector/
├── app.py                  # Streamlit web app
├── train.py                # YOLOv8 fine-tuning script
├── prepare_codebrim.py     # Dataset conversion to YOLO format
├── requirements.txt
├── utils/
│   ├── visualiser.py       # Bounding box drawing and annotation
│   └── metrics.py          # Detection summary stats
├── models/
│   └── .gitkeep            # Place fine-tuned .pt weights here
└── data/
    └── sample_images/      # Add sample bridge images for testing
```

---

## Results

| Metric | Value |
|---|---|
| Dataset | CODEBRIM (6 classes) |
| Base Model | YOLOv8n |
| Epochs | 50 |
| Image Size | 640px |
| mAP@0.5 | *Update after training* |
| Inference Speed | ~30ms/image (CPU) |

---

## Relevance to STRUCTURE & MariSens Projects

The [STRUCTURE](https://structure-project.eu) EU project develops AI-enabled UAV inspection and digital twin systems for transportation infrastructure. This project demonstrates the core AI perception capability that feeds into a digital twin pipeline:

```
UAV capture → AI defect detection → World-frame localisation → Digital twin update
```

The same approach extends to the [MariSens](https://marisens-project.eu) project for offshore and coastal maritime asset inspection.

---

## Author

**Lakshan Divakar**
MSc Electronics & Electrical Engineering, Brunel University London
Dissertation: *Autonomous UAV Infrastructure Inspection using LiDAR and YOLOv5*

[GitHub](https://github.com/lakshan-d) | [LinkedIn](https://linkedin.com/in/lakshan-d) | [Email](mailto:lakshan.d.2108@gmail.com)

---

## Dataset Credit

Mundt, M., Majumder, S., Murali, S., Panetsos, P., & Ramesh, V. (2019).
**Meta-learning convolutional neural architectures for multi-target concrete defect classification with the COncrete DEfect BRidge IMage dataset.**
CVPR 2019. [Zenodo](https://zenodo.org/record/2579133)

---

## License

MIT License — free to use, modify, and share with attribution.
