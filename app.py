import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import time
from utils.visualiser import draw_detections, generate_report_data
from utils.metrics import compute_summary

st.set_page_config(
    page_title="Bridge Defect Detection System",
    page_icon="🌉",
    layout="wide"
)

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1F3863; }
    .sub-title  { font-size: 1rem; color: #595959; margin-bottom: 1.5rem; }
    .metric-card {
        background: #f0f4ff; border-radius: 10px;
        padding: 1rem; text-align: center; border-left: 4px solid #1F3863;
    }
    .defect-tag {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 0.8rem; font-weight: 600; margin: 2px;
    }
    .footer { font-size: 0.75rem; color: #999; text-align: center; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🌉 Bridge Defect Detection System</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">AI-powered infrastructure inspection using YOLOv8 — '
    'aligned with the EU STRUCTURE project on UAV-based bridge inspection and digital twin development.</p>',
    unsafe_allow_html=True
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Detection Settings")

    confidence = st.slider(
        "Confidence Threshold", 0.1, 0.95, 0.35, 0.05,
        help="Lower = more detections, higher = more certain detections"
    )

    iou_thresh = st.slider(
        "IoU Threshold (NMS)", 0.1, 0.9, 0.45, 0.05,
        help="Controls overlap suppression between bounding boxes"
    )

    show_labels   = st.checkbox("Show Labels",      value=True)
    show_conf     = st.checkbox("Show Confidence",  value=True)
    show_boxes    = st.checkbox("Show Bounding Boxes", value=True)

    st.divider()
    st.markdown("**Defect Classes (CODEBRIM)**")
    defect_info = {
        "Crack":           "#e74c3c",
        "Spalling":        "#e67e22",
        "Corrosion":       "#8e44ad",
        "Efflorescence":   "#2980b9",
        "Exposed Rebar":   "#27ae60",
        "Background":      "#95a5a6",
    }
    for defect, colour in defect_info.items():
        st.markdown(
            f'<span class="defect-tag" style="background:{colour};color:white">{defect}</span>',
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown("**About**")
    st.caption(
        "Built by Lakshan Divakar. MSc Brunel University London. "
        "Dataset: CODEBRIM (Concrete Bridge Defect). "
        "Model: YOLOv8 fine-tuned for infrastructure inspection."
    )

# ── Model loader ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load YOLOv8 model. Uses fine-tuned weights if available, else pretrained."""
    model_path = "models/bridge_defect_yolov8.pt"
    if os.path.exists(model_path):
        model = YOLO(model_path)
        st.sidebar.success("Loaded fine-tuned model")
    else:
        model = YOLO("yolov8n.pt")
        st.sidebar.info("Using base YOLOv8n (fine-tune with CODEBRIM for best results)")
    return model

model = load_model()

# ── Input mode ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Image Upload", "Video Upload", "About & Usage"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — IMAGE
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Upload Bridge Inspection Image")
    uploaded_file = st.file_uploader(
        "Upload an image (JPG, PNG, JPEG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a photo of a bridge surface, deck, or structural element"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Original Image**")
            st.image(image, use_container_width=True)

        with st.spinner("Running defect detection..."):
            start = time.time()
            results = model.predict(
                img_array,
                conf=confidence,
                iou=iou_thresh,
                verbose=False
            )
            elapsed = time.time() - start

        annotated = draw_detections(
            img_array.copy(), results,
            show_labels=show_labels,
            show_conf=show_conf,
            show_boxes=show_boxes
        )

        with col2:
            st.markdown("**Detected Defects**")
            st.image(annotated, use_container_width=True)

        # Metrics
        st.divider()
        summary = compute_summary(results)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Detections", summary["total"])
        m2.metric("Inference Time",   f"{elapsed:.2f}s")
        m3.metric("Avg Confidence",   f"{summary['avg_conf']:.1%}" if summary["avg_conf"] else "N/A")
        m4.metric("Defect Types",     summary["unique_classes"])

        # Per-class breakdown
        if summary["class_counts"]:
            st.markdown("**Defect Breakdown**")
            for cls, count in summary["class_counts"].items():
                colour = defect_info.get(cls, "#333")
                st.markdown(
                    f'<span class="defect-tag" style="background:{colour};color:white">'
                    f'{cls}: {count}</span>',
                    unsafe_allow_html=True
                )

        # Download annotated image
        st.divider()
        annotated_pil = Image.fromarray(annotated)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            annotated_pil.save(tmp.name, quality=95)
            with open(tmp.name, "rb") as f:
                st.download_button(
                    "Download Annotated Image",
                    data=f.read(),
                    file_name="bridge_defect_result.jpg",
                    mime="image/jpeg"
                )

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — VIDEO
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Upload Bridge Inspection Video")
    st.info("Upload a short video (MP4, AVI, MOV). Each frame will be analysed for defects.")
    video_file = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov"],
        help="Simulates UAV-captured video analysis of bridge structures"
    )

    if video_file:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_vid:
            tmp_vid.write(video_file.read())
            tmp_path = tmp_vid.name

        cap    = cv2.VideoCapture(tmp_path)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        st.info(f"Video: {total} frames | {fps:.1f} FPS | {width}x{height}")

        out_path = tmp_path.replace(".mp4", "_annotated.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        progress  = st.progress(0)
        frame_ph  = st.empty()
        all_dets  = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb, conf=confidence, iou=iou_thresh, verbose=False)
            annotated_frame = draw_detections(rgb.copy(), results, show_labels, show_conf, show_boxes)

            bgr_out = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr_out)

            all_dets.append(compute_summary(results))

            if frame_idx % 10 == 0:
                frame_ph.image(annotated_frame, caption=f"Frame {frame_idx}/{total}", use_container_width=True)
                progress.progress(min(frame_idx / max(total, 1), 1.0))

            frame_idx += 1

        cap.release()
        writer.release()
        progress.progress(1.0)

        st.success(f"Processed {frame_idx} frames.")

        total_dets = sum(d["total"] for d in all_dets)
        frames_with_defects = sum(1 for d in all_dets if d["total"] > 0)
        st.metric("Total Defect Detections (all frames)", total_dets)
        st.metric("Frames Containing Defects", f"{frames_with_defects} / {frame_idx}")

        with open(out_path, "rb") as f:
            st.download_button(
                "Download Annotated Video",
                data=f.read(),
                file_name="bridge_inspection_annotated.mp4",
                mime="video/mp4"
            )

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("About This Project")
    st.markdown("""
This system demonstrates **AI-enabled automated defect detection** for bridge and civil infrastructure inspection,
directly aligned with the EU-funded **STRUCTURE project** on UAV-based inspection and digital twin development.

### What it does
- Detects six defect classes on concrete bridge surfaces: **Crack, Spalling, Corrosion, Efflorescence, Exposed Rebar, Background**
- Runs inference on uploaded images or video frames
- Outputs annotated results with bounding boxes, class labels, and confidence scores
- Simulates the perception layer of a UAV inspection pipeline

### Dataset
Trained (or fine-tunable) on **CODEBRIM** — Concrete Bridge Defect Recognition dataset.
[Download CODEBRIM here](https://zenodo.org/record/2579!3/files/CODEBRIM.zip)

### Tech Stack
- **Model:** YOLOv8 (Ultralytics)
- **Framework:** Python, OpenCV, Streamlit
- **Training:** PyTorch, Google Colab (GPU)
- **Deployment:** Streamlit Cloud or local

### How to fine-tune on CODEBRIM
```bash
python train.py --data data/codebrim.yaml --epochs 50 --weights yolov8n.pt
```

### Relevance to STRUCTURE Project
The STRUCTURE EU project develops AI-enabled UAV inspection and digital twin systems for transportation
infrastructure with a focus on bridges. This tool demonstrates the core perception capability:
automated defect localisation from visual inspection data.

### Author
**Lakshan Divakar** — MSc Electronics & Electrical Engineering, Brunel University London  
[GitHub](https://github.com/lakshan-d) | [LinkedIn](https://linkedin.com/in/lakshan-d)
    """)

st.markdown('<p class="footer">Bridge Defect Detection System — Lakshan Divakar — Brunel University London</p>', unsafe_allow_html=True)
