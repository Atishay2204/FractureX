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
    --page: #0B1220;
    --card: #111C2D;
    --ink: #E6EDF5;
    --muted: #8FA3B8;
    --line: #223249;
    --panel: #050B16;
    --accent: #2BB5C0;
    --signal: #F2B13A;
    --ok: #4CC38A;
}

html, body, [class*="css"], .stApp {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
    color: var(--ink);
}
.stApp { background: var(--page); }
.block-container { padding-top: 2.2rem; max-width: 1280px; }
header[data-testid="stHeader"] { background: transparent; }

/* Dark-mode text and widget fixes */
h1, h2, h3, h4, label, p, li, span, [data-testid="stMarkdownContainer"] { color: var(--ink); }
.stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"] { color: var(--muted); }
[data-testid="stExpander"] { background: var(--card); border: 1px solid var(--line); border-radius: 10px; }
[data-testid="stFileUploader"] small, [data-testid="stCaptionContainer"] { color: var(--muted); }
.stDownloadButton > button { background: var(--card); color: var(--ink); border: 1px solid var(--line); }
.stDownloadButton > button:hover { border-color: var(--accent); color: var(--accent); }

/* Self-contained dark theme: no config.toml needed */
:root, .stApp { color-scheme: dark; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--page) !important; }
[data-testid="stSidebar"] > div:first-child { background: #0F1829 !important; }
[data-testid="stSidebar"] * { color: var(--ink); }
[data-testid="stToolbar"], [data-testid="stHeader"] { background: transparent !important; }
[data-testid="stFileUploader"] section > div, [data-testid="stFileUploaderDropzoneInstructions"] * { color: var(--muted) !important; }
[data-testid="stFileUploader"] button { background: var(--panel-btn, #1A2A42); color: var(--ink); border: 1px solid var(--line); }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: var(--ink) !important; }
[data-testid="stAlert"] { background: var(--card); border: 1px solid var(--line); border-radius: 10px; }
[data-testid="stAlert"] * { color: var(--ink) !important; }
[data-testid="stSpinner"] * { color: var(--muted) !important; }
[data-testid="stSlider"] [role="slider"] { background: var(--accent) !important; }
[data-testid="stSlider"] [data-baseweb="slider"] > div > div { background: #2A3D57; }
[data-testid="stSlider"] [data-testid="stThumbValue"] { color: var(--accent) !important; }
.stButton > button[kind="primary"], .stButton > button[kind="primary"] * { color: #04222A !important; }
hr { border-color: var(--line) !important; }

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
    border: 1px solid var(--line);
    padding: .6rem;
    border-radius: 14px;
}
[data-testid="stImageCaption"] { color: var(--muted) !important; font-size: .8rem; }

/* Uploader */
[data-testid="stFileUploader"] section {
    background: var(--card);
    border: 1.5px dashed #3A4F68;
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
    background: var(--accent); border: none; color: #04222A;
}
.stButton > button[kind="primary"]:hover { background: #22A0AB; }
.stButton > button:active, .stDownloadButton > button:active { transform: scale(.98); }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
    outline: 3px solid var(--signal); outline-offset: 2px;
}

/* Stat row */
.stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: .7rem; margin: .2rem 0 1rem 0; }
.stat { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: .8rem 1rem; }
.stat .v { font-family: 'Newsreader', Georgia, serif; font-size: 1.9rem; font-weight: 600; line-height: 1.1; }
.stat .l { color: var(--muted); font-size: .8rem; margin-top: .15rem; }
.stat.signal { border-left: 4px solid var(--signal); }
.stat.clear { border-left: 4px solid var(--ok); }

/* Verdict banner */
.verdict { border-radius: 12px; padding: .9rem 1.1rem; margin-bottom: .9rem; font-weight: 500; }
.verdict.signal { background: rgba(242,177,58,.12); border: 1px solid rgba(242,177,58,.45); }
.verdict.clear { background: rgba(76,195,138,.12); border: 1px solid rgba(76,195,138,.45); }
.verdict small { display: block; font-weight: 400; color: var(--muted); margin-top: .15rem; }

/* Finding cards */
.finding { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: .8rem 1rem; margin-bottom: .55rem; }
.finding .top { display: flex; justify-content: space-between; align-items: baseline; gap: .5rem; }
.finding .name { font-weight: 600; }
.finding .pct { font-variant-numeric: tabular-nums; font-weight: 600; }
.finding .tag { font-size: .78rem; color: var(--muted); }
.bar { height: 6px; background: #223249; border-radius: 99px; margin-top: .55rem; overflow: hidden; }
.bar > span { display: block; height: 100%; border-radius: 99px; }
.bar > span.hi { background: var(--signal); }
.bar > span.mid { background: #C8963A; }
.bar > span.lo { background: #5B7089; }

/* Empty state */
.empty {
    background: var(--card); border: 1px solid var(--line); border-radius: 14px;
    padding: 1.4rem 1.5rem; color: var(--muted);
}
.empty h4 { margin: 0 0 .4rem 0; color: var(--ink); font-size: 1rem; }
.empty ol { margin: .4rem 0 0 1.1rem; padding: 0; }
.empty li { margin: .25rem 0; }

/* Sidebar */
[data-testid="stSidebar"] { background: #0F1829; border-right: 1px solid var(--line); }
.disclaimer {
    background: var(--card); border-left: 4px solid var(--signal); border-radius: 8px;
    padding: .8rem .9rem; font-size: .83rem; color: var(--ink);
}

.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: .7rem; margin: 1.2rem 0 .4rem 0; }
.step { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: .75rem 1rem; }
.step b { display: block; margin-bottom: .1rem; }
.step span { color: var(--muted); font-size: .85rem; }
.next { background: var(--card); border: 1px solid var(--line); border-left: 4px solid var(--accent);
        border-radius: 10px; padding: .85rem 1rem; margin-top: .9rem; font-size: .92rem; }
.next b { display: block; margin-bottom: .2rem; }
.privacy { color: var(--muted); font-size: .8rem; margin-top: .6rem; }
@media (max-width: 760px) { .steps { grid-template-columns: 1fr; } }
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
        return "hi", "Likely"
    if pct >= 40:
        return "mid", "Possible"
    return "lo", "Unsure, worth a second look"


def to_png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
# Settings (plain language; the technical threshold stays hidden)
# ─────────────────────────────────────────────────────────────
SENSITIVITY = {
    "Strict: fewer alerts": 50,
    "Balanced (recommended)": 25,
    "Sensitive: more alerts": 15,
}
MAX_MB = 10

with st.sidebar:
    st.markdown("### About this tool")
    st.markdown(
        "This tool looks at an X-ray image and highlights areas that may show a broken bone. "
        "It checks nine body regions."
    )
    with st.expander("Which body parts are covered?"):
        for v in CLASS_NAMES.values():
            st.markdown(f"- {v.replace(' fracture', '')}")
    with st.expander("What it cannot do"):
        st.markdown(
            "- It cannot replace a doctor.\n"
            "- It can miss small or unusual fractures.\n"
            "- It can highlight healthy bone by mistake.\n"
            "- It does not check for other conditions."
        )
    with st.expander("Advanced settings"):
        level = st.select_slider(
            "How cautious should the check be?",
            options=list(SENSITIVITY.keys()),
            value="Balanced (recommended)",
            help="Sensitive shows more possible fractures but also more false alarms. "
            "Strict shows fewer, but may miss subtle ones.",
        )
    confidence = SENSITIVITY[level]
    st.markdown(
        '<div class="disclaimer"><b>For education and demonstration only.</b> '
        "Only a qualified doctor can diagnose a fracture.</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
  <h1>Bone fracture check</h1>
  <p>Upload an X-ray image. The tool marks areas that may show a broken bone and tells you where.</p>
</div>
<div class="steps">
  <div class="step"><b>1. Upload</b><span>Choose an X-ray image from your device.</span></div>
  <div class="step"><b>2. Wait a few seconds</b><span>The check starts on its own.</span></div>
  <div class="step"><b>3. Review</b><span>See the marked areas and share them with your doctor.</span></div>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")

# ─────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<p class="panel-title">Your X-ray</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload an X-ray (JPG or PNG, up to 10 MB)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        help="JPG or PNG, up to 10 MB.",
    )
    with st.expander("Tips for a better result"):
        st.markdown(
            "- Use a clear X-ray, not a blurry or cropped photo.\n"
            "- Make sure the whole bone and the joints beside it are in the picture.\n"
            "- If you photograph a film, avoid glare and shadows.\n"
            "- Upload one body part at a time."
        )
    st.markdown(
        '<p class="privacy">This app does not save your image.</p>',
        unsafe_allow_html=True,
    )

    image, load_error = None, None
    if uploaded_file:
        if uploaded_file.size > MAX_MB * 1024 * 1024:
            load_error = f"This file is larger than {MAX_MB} MB. Please upload a smaller image."
        else:
            try:
                image = Image.open(uploaded_file).convert("RGB")
            except Exception:
                load_error = "We could not open this file. Please upload a valid JPG or PNG image."
        if load_error:
            st.error(load_error)
        else:
            st.image(image, caption=uploaded_file.name, use_container_width=True)
    else:
        st.session_state.pop("result", None)
        st.session_state.pop("run_key", None)

# The check runs automatically when a new image is uploaded or the setting changes
analysis_error = None
if image is not None:
    run_key = (uploaded_file.name, uploaded_file.size, confidence)
    if st.session_state.get("run_key") != run_key:
        with right:
            with st.spinner("Checking your X-ray. This takes a few seconds..."):
                try:
                    start = time.perf_counter()
                    results = model.predict(np.array(image), conf=confidence / 100, verbose=False)
                    elapsed = time.perf_counter() - start
                    annotated = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
                    rows = [
                        {
                            "Region": CLASS_NAMES.get(int(b.cls[0]), f"Class {int(b.cls[0])}"),
                            "Confidence": float(b.conf[0]) * 100,
                        }
                        for b in results[0].boxes
                    ]
                    rows.sort(key=lambda r: r["Confidence"], reverse=True)
                    st.session_state["result"] = {
                        "file": uploaded_file.name,
                        "image": annotated,
                        "rows": rows,
                        "secs": elapsed,
                    }
                    st.session_state["run_key"] = run_key
                except Exception:
                    st.session_state.pop("result", None)
                    st.session_state.pop("run_key", None)
                    analysis_error = "Something went wrong while checking this image. Please try again or use a different image."

with right:
    st.markdown('<p class="panel-title">Results</p>', unsafe_allow_html=True)
    result = st.session_state.get("result")

    if analysis_error:
        st.error(analysis_error)
    elif image is None or not result:
        st.markdown(
            """
<div class="empty">
  <h4>Your results will appear here</h4>
  Upload an X-ray on the left. The check starts automatically.
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        rows = result["rows"]
        n = len(rows)

        if n == 0:
            st.markdown(
                '<div class="verdict clear">No fracture found in this image.'
                "<small>This does not rule out a fracture. If you are in pain, see a doctor.</small></div>",
                unsafe_allow_html=True,
            )
        else:
            regions = ", ".join(dict.fromkeys(r["Region"].replace(" fracture", "").lower() for r in rows))
            st.markdown(
                f'<div class="verdict signal">Possible fracture found: {regions}.'
                "<small>Look at the marked areas below and show this to a doctor.</small></div>",
                unsafe_allow_html=True,
            )

        st.image(
            result["image"],
            caption="Marked areas show where the tool sees a possible fracture.",
            use_container_width=True,
        )

        if n:
            st.markdown('<p class="panel-title" style="margin-top:1rem">What it found</p>', unsafe_allow_html=True)
            for r in rows:
                band, label = confidence_band(r["Confidence"])
                st.markdown(
                    f"""
<div class="finding">
  <div class="top">
    <span class="name">{r['Region']}</span>
    <span class="tag">{label} · {r['Confidence']:.0f}%</span>
  </div>
  <div class="bar"><span class="{band}" style="width:{r['Confidence']:.0f}%"></span></div>
</div>
""",
                    unsafe_allow_html=True,
                )

        st.markdown(
            """
<div class="next"><b>What to do next</b>
Show this result and your original X-ray to a doctor or radiologist.
Do not use it to decide on treatment.</div>
""",
            unsafe_allow_html=True,
        )

        st.write("")
        stem = os.path.splitext(result["file"])[0]
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Save marked image",
                data=to_png_bytes(result["image"]),
                file_name=f"{stem}_marked.png",
                mime="image/png",
                use_container_width=True,
            )
        with d2:
            if n:
                df = pd.DataFrame(rows).round({"Confidence": 1})
                st.download_button(
                    "Save results (CSV)",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{stem}_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )