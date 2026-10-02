import io
import os
import time
import urllib.request
import zipfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fracture Screening",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Styling: a radiology "light box" feel.
# Pale clinical page, dark viewing panels for the images,
# and one signal colour (amber) reserved for findings.
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');

:root {
    --page: #F2F5F7;
    --ink: #12202F;
    --muted: #5B6B7B;
    --line: #D5DDE4;
    --panel: #0A1628;
    --panel-2: #14243A;
    --accent: #0E7C86;
    --signal: #E39B14;
    --ok: #2F8F5B;
}

html, body, [class*="css"], .stApp {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
    color: var(--ink);
}
.stApp { background: var(--page); }
.block-container { padding-top: 2.2rem; max-width: 1280px; }
header[data-testid="stHeader"] { background: transparent; }

/* Title block */
.hero h1 {
    font-family: 'Newsreader', Georgia, serif;
    font-weight: 600;
    font-size: 2.9rem;
    letter-spacing: -0.02em;
    line-height: 1.05;
    margin: 0 0 .5rem 0;
    padding: 0;
    color: var(--ink);
}
.hero p {
    color: var(--muted);
    font-size: 1.05rem;
    max-width: 62ch;
    margin: 0;
}
.hero-meta {
    display: flex; gap: 1.4rem; flex-wrap: wrap;
    margin-top: 1.1rem; font-size: .85rem; color: var(--muted);
}
.hero-meta b { color: var(--ink); font-weight: 600; }

/* Panel headings */
.panel-title {
    font-weight: 600; font-size: 1rem; margin: 0 0 .6rem 0; color: var(--ink);
}
.panel-note { color: var(--muted); font-size: .85rem; margin: -.3rem 0 .8rem 0; }

/* Dark viewing panels hold the images */
[data-testid="stImage"] img {
    border-radius: 10px;
    background: var(--panel);
}
[data-testid="stImage"] {
    background: var(--panel);
    padding: .6rem;
    border-radius: 14px;
}
[data-testid="stImageCaption"] { color: #9FB3C8 !important; font-size: .8rem; }

/* Uploader */
[data-testid="stFileUploader"] section {
    background: #fff;
    border: 1.5px dashed #9FB3C8;
    border-radius: 12px;
    padding: 1.6rem;
}
[data-testid="stFileUploader"] section:hover { border-color: var(--accent); }

/* Buttons */
.stButton > button, .stDownloadButton > button {
    border-radius: 10px; font-weight: 600; padding: .65rem 1rem;
    transition: transform .08s ease;
}
.stButton > button[kind="primary"] {
    background: var(--accent); border: none; color: #fff;
}
.stButton > button[kind="primary"]:hover { background: #0B646C; }
.stButton > button:active, .stDownloadButton > button:active { transform: scale(.98); }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
    outline: 3px solid var(--signal); outline-offset: 2px;
}

/* Stat row */
.stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: .7rem; margin: .2rem 0 1rem 0; }
.stat { background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: .8rem 1rem; }
.stat .v { font-family: 'Newsreader', Georgia, serif; font-size: 1.9rem; font-weight: 600; line-height: 1.1; }
.stat .l { color: var(--muted); font-size: .8rem; margin-top: .15rem; }
.stat.signal { border-left: 4px solid var(--signal); }
.stat.clear { border-left: 4px solid var(--ok); }

/* Verdict banner */
.verdict { border-radius: 12px; padding: .9rem 1.1rem; margin-bottom: .9rem; font-weight: 500; }
.verdict.signal { background: #FFF5E0; border: 1px solid #F1D08A; }
.verdict.clear { background: #E8F5EE; border: 1px solid #A9D6BC; }
.verdict small { display: block; font-weight: 400; color: var(--muted); margin-top: .15rem; }

/* Finding cards */
.finding { background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: .8rem 1rem; margin-bottom: .55rem; }
.finding .top { display: flex; justify-content: space-between; align-items: baseline; gap: .5rem; }
.finding .name { font-weight: 600; }
.finding .pct { font-variant-numeric: tabular-nums; font-weight: 600; }
.finding .tag { font-size: .78rem; color: var(--muted); }
.bar { height: 6px; background: #E6ECF1; border-radius: 99px; margin-top: .55rem; overflow: hidden; }
.bar > span { display: block; height: 100%; border-radius: 99px; }
.bar > span.hi { background: var(--signal); }
.bar > span.mid { background: #EFC365; }
.bar > span.lo { background: #B7C4D1; }

/* Empty state */
.empty {
    background: #fff; border: 1px solid var(--line); border-radius: 14px;
    padding: 1.4rem 1.5rem; color: var(--muted);
}
.empty h4 { margin: 0 0 .4rem 0; color: var(--ink); font-size: 1rem; }
.empty ol { margin: .4rem 0 0 1.1rem; padding: 0; }
.empty li { margin: .25rem 0; }

/* Sidebar */
[data-testid="stSidebar"] { background: #E7EDF2; border-right: 1px solid var(--line); }
.disclaimer {
    background: #fff; border-left: 4px solid var(--signal); border-radius: 8px;
    padding: .8rem .9rem; font-size: .83rem; color: var(--ink);
}

@media (max-width: 760px) {
    .hero h1 { font-size: 2.1rem; }
    .stats { grid-template-columns: 1fr; }
}
@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────
MODEL_URL = "https://github.com/Atishay2204/bone-fracture-detection/releases/download/v1.0/best.zip"


@st.cache_resource
def load_model():
    if not os.path.exists("best.pt"):
        with st.spinner("Downloading model weights. This only happens on the first run."):
            urllib.request.urlretrieve(MODEL_URL, "best_downloaded.zip")
            with zipfile.ZipFile("best_downloaded.zip", "r") as zf:
                pt_files = [f for f in zf.namelist() if f.endswith(".pt")]
                if pt_files:
                    zf.extractall(".")
            if pt_files:
                os.remove("best_downloaded.zip")
                for root, _, files in os.walk("."):
                    for f in files:
                        if f.endswith(".pt"):
                            return YOLO(os.path.join(root, f))
            else:
                # The download itself is the PyTorch weights file (.pt files are zip archives)
                os.rename("best_downloaded.zip", "best.pt")
    return YOLO("best.pt")


model = load_model()

CLASS_NAMES = {
    0: "Elbow fracture",
    1: "Finger fracture",
    2: "Forearm fracture",
    3: "Humerus fracture",
    4: "Shoulder fracture",
    5: "Wrist fracture",
    6: "Hand fracture",
    7: "Hip fracture",
    8: "Leg fracture",
}


def confidence_band(pct: float):
    if pct >= 70:
        return "hi", "High confidence"
    if pct >= 40:
        return "mid", "Moderate confidence"
    return "lo", "Low confidence, review closely"


def to_png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
  <h1>Fracture screening for X-rays</h1>
  <p>Upload a scan and the model marks where it sees a possible fracture and names the region.
  Results are a second look, not a diagnosis.</p>
  <div class="hero-meta">
    <span><b>YOLOv8l</b> detector</span>
    <span><b>9</b> body regions</span>
    <span><b>15,000+</b> training images</span>
    <span><b>51.5%</b> mAP50</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    confidence = st.slider(
        "Minimum confidence (%)",
        min_value=10,
        max_value=90,
        value=25,
        step=5,
        help="Lower shows more possible fractures but adds false alarms. Higher shows fewer, more certain ones.",
    )
    st.caption("Change this, then select **Analyze scan** again to apply it.")

    with st.expander("Regions the model covers"):
        for v in CLASS_NAMES.values():
            st.markdown(f"- {v.replace(' fracture', '')}")

    st.write("")
    st.markdown(
        '<div class="disclaimer"><b>For education and demonstration only.</b> '
        "A qualified radiologist must confirm any clinical finding.</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<p class="panel-title">1. Scan</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload an X-ray (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    image = None
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"{uploaded_file.name} · {image.width}×{image.height}px", use_container_width=True)
        analyze = st.button("Analyze scan", type="primary", use_container_width=True)
    else:
        analyze = False
        # Clear any old result when the file is removed
        st.session_state.pop("result", None)

# Run inference and keep the result, so it survives reruns (e.g. moving the slider)
if uploaded_file and analyze:
    with right:
        with st.spinner("Analyzing scan..."):
            start = time.perf_counter()
            results = model.predict(np.array(image), conf=confidence / 100, verbose=False)
            elapsed_ms = (time.perf_counter() - start) * 1000

            annotated = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
            rows = []
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                rows.append(
                    {
                        "Region": CLASS_NAMES.get(class_id, f"Class {class_id}"),
                        "Confidence": float(box.conf[0]) * 100,
                    }
                )
            rows.sort(key=lambda r: r["Confidence"], reverse=True)

    st.session_state["result"] = {
        "file": uploaded_file.name,
        "image": annotated,
        "rows": rows,
        "ms": elapsed_ms,
        "threshold": confidence,
    }

with right:
    st.markdown('<p class="panel-title">2. Findings</p>', unsafe_allow_html=True)
    result = st.session_state.get("result")

    if not uploaded_file or not result:
        st.markdown(
            """
<div class="empty">
  <h4>No scan analyzed yet</h4>
  <ol>
    <li>Upload an X-ray on the left.</li>
    <li>Set the minimum confidence in the sidebar if you want to change it.</li>
    <li>Select <b>Analyze scan</b>.</li>
  </ol>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        rows = result["rows"]
        n = len(rows)

        # Verdict banner
        if n == 0:
            st.markdown(
                f'<div class="verdict clear">No fractures found at {result["threshold"]}% confidence or higher.'
                "<small>Try a lower threshold if you suspect a subtle fracture.</small></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="verdict signal">{n} possible fracture{"s" if n != 1 else ""} marked.'
                "<small>Review each box on the image below.</small></div>",
                unsafe_allow_html=True,
            )

        # Stats
        top = f'{rows[0]["Confidence"]:.0f}%' if n else "–"
        st.markdown(
            f"""
<div class="stats">
  <div class="stat {'signal' if n else 'clear'}"><div class="v">{n}</div><div class="l">Findings</div></div>
  <div class="stat"><div class="v">{top}</div><div class="l">Highest confidence</div></div>
  <div class="stat"><div class="v">{result['ms']/1000:.1f}s</div><div class="l">Analysis time</div></div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.image(result["image"], caption="Model output. Boxes show suspected fracture locations.", use_container_width=True)

        if result["threshold"] != confidence:
            st.info(f"Showing results at {result['threshold']}%. Select Analyze scan again to use {confidence}%.")

        # Finding cards
        if n:
            st.write("")
            for r in rows:
                band, label = confidence_band(r["Confidence"])
                st.markdown(
                    f"""
<div class="finding">
  <div class="top">
    <span class="name">{r['Region']}</span>
    <span class="pct">{r['Confidence']:.1f}%</span>
  </div>
  <div class="tag">{label}</div>
  <div class="bar"><span class="{band}" style="width:{r['Confidence']:.0f}%"></span></div>
</div>
""",
                    unsafe_allow_html=True,
                )

        # Downloads
        st.write("")
        d1, d2 = st.columns(2)
        stem = os.path.splitext(result["file"])[0]
        with d1:
            st.download_button(
                "Download annotated image",
                data=to_png_bytes(result["image"]),
                file_name=f"{stem}_annotated.png",
                mime="image/png",
                use_container_width=True,
            )
        with d2:
            if n:
                df = pd.DataFrame(rows).round({"Confidence": 1})
                st.download_button(
                    "Download findings (CSV)",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{stem}_findings.csv",
                    mime="text/csv",
                    use_container_width=True,
                )