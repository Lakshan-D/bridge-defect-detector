 # 🌉 Bridge Defect Detection System

**AI-powered automated defect detection for bridge and civil infrastructure inspection — aligned with the EU-funded [STRUCTURE project](https://structure-project.eu) on UAV-based inspection and digital twin development.**

---

## Demo

> Upload a bridge inspection image or video. The system detects and localises defects in real time.

![Sample Detection](docs/sample_detection.jpg)

---

## What It Does

This system uses a fine-tuned **YOLOv8** model to automatically detect and localise cracks in concrete bridge surfaces from inspection images or UAV-captured video frames.

**Detects crack defects** in concrete infrastructure surfaces, demonstrating the core AI perception layer of a UAV-based inspection pipeline.

---

## Why This Matters

Manual bridge inspection is slow, expensive, and dangerous. Automating defect detection from UAV-captured imagery is a core challenge in modern infrastructure maintenance. This project demonstrates the **AI perception layer** of a UAV inspection pipeline — the step between raw image capture and structured digital twin output.

This work directly mirrors the technical approach of the EU STRUCTURE project, which develops AI-enabled UAV inspection and digital twin systems for transportation infrastructure with a focus on bridges.

---

## Tech Stack

- **Model:** YOLOv8 (Ultralytics) — trained on synthetic concrete crack dataset
- **App:** Streamlit
- **Vision:** OpenCV, Pillow
- **Training:** PyTorch (GPU via Google Colab T4)
- **Language:** Python 3.10+

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/Lakshan-D/bridge-defect-detector.git
cd bridge-defect-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app runs in your browser at `http://localhost:8501`. Place fine-tuned weights in `models/bridge_defect_yolov8.pt` for best results.

---

## Training Your Own Model

Use the included Colab notebook `train_codebrim.ipynb` to fine-tune on your own dataset.

```bash
python train.py
```

Fine-tuned weights are saved automatically to `models/bridge_defect_yolov8.pt` and loaded by the app automatically.

---

## Project Structure

```
bridge-defect-detector/
├── app.py                  # Streamlit web app
├── train.py                # YOLOv8 fine-tuning script
├── train_codebrim.ipynb    # Google Colab training notebook
├── prepare_codebrim.py     # Dataset preparation script
├── requirements.txt
├── utils/
│   ├── visualiser.py       # Bounding box drawing and annotation
│   └── metrics.py          # Detection summary stats
├── models/
│   └── .gitkeep            # Place fine-tuned .pt weights here
└── docs/
    └── sample_detection.jpg
```

---

## Results

| Metric | Value |
|---|---|
| Dataset | Synthetic concrete crack dataset (1,000 images) |
| Base Model | YOLOv8n |
| Epochs | 50 |
| Image Size | 640px |
| mAP@0.5 | **0.991 (99.1%)** |
| mAP@0.5:0.95 | **0.920 (92.0%)** |
| Precision | **0.990 (99.0%)** |
| Recall | **0.952 (95.2%)** |
| Inference Speed | ~7ms/image (T4 GPU) |

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

[GitHub](https://github.com/Lakshan-D) | [LinkedIn](https://linkedin.com/in/lakshan-d) | [Email](mailto:lakshan.d.2108@gmail.com)

---

## License

MIT License — free to use, modify, and share with attribution.
