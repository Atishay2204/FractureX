import io
import os
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────
# Page config  (NO sidebar)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OsteoScan AI · Fracture Detection",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# Premium dark UI — single-page, no sidebar
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg-base:      #060a10;
    --bg-surface:   #0b1219;
    --bg-card:      #101b28;
    --bg-elevated:  #152235;
    --bg-glass:     rgba(16,27,40,0.78);
    --border:       rgba(99,152,220,0.12);
    --border-hover: rgba(99,152,220,0.30);
    --border-accent:rgba(62,207,218,0.25);
    --ink:          #e0ecf7;
    --ink-secondary:#b0c8de;
    --ink-muted:    #5f84a3;
    --ink-faint:    #2d4560;
    --accent:       #3ecfda;
    --accent-glow:  rgba(62,207,218,0.16);
    --accent-deep:  #1b9faa;
    --accent-soft:  rgba(62,207,218,0.08);
    --warn:         #f5a623;
    --warn-glow:    rgba(245,166,35,0.12);
    --warn-soft:    rgba(245,166,35,0.08);
    --ok:           #34d399;
    --ok-glow:      rgba(52,211,153,0.10);
    --danger:       #ff5f72;
    --radius-xs:    8px;
    --radius-sm:    12px;
    --radius-md:    16px;
    --radius-lg:    24px;
    --radius-xl:    32px;
    --shadow-sm:    0 2px 8px rgba(0,0,0,0.24);
    --shadow-md:    0 4px 24px rgba(0,0,0,0.40);
    --shadow-lg:    0 8px 48px rgba(0,0,0,0.55);
    --shadow-glow:  0 0 40px var(--accent-glow);
}

*, *::before, *::after { box-sizing: border-box; }
html, body { color-scheme: dark; }
html, body, .stApp, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background: var(--bg-base) !important;
    color: var(--ink);
}

.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--bg-base) !important; }
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
    background: transparent !important; border: none !important;
}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }

.block-container {
    padding: 1rem 2.5rem 4rem;
    max-width: 1420px;
    margin: 0 auto;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--ink-faint); border-radius: 99px; }

/* ── Top Nav ── */
.topnav {
    display: flex; align-items: center; justify-content: space-between;
    padding: .8rem 0; margin-bottom: 1rem; border-bottom: 1px solid var(--border);
}
.topnav-left { display: flex; align-items: center; gap: .7rem; }
.topnav-logo {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, var(--accent), var(--accent-deep));
    border-radius: var(--radius-xs);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.15rem; box-shadow: 0 0 20px var(--accent-glow);
}
.topnav-brand {
    font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem;
    font-weight: 700; letter-spacing: -.02em; color: var(--ink);
}
.topnav-brand span { color: var(--accent); }
.topnav-right { display: flex; align-items: center; gap: 1.4rem; }
.topnav-link { font-size: .82rem; font-weight: 500; color: var(--ink-muted); cursor: default; }
.topnav-badge {
    display: inline-flex; align-items: center; gap: .35rem;
    background: var(--accent-soft); border: 1px solid var(--border-accent);
    border-radius: 99px; padding: .22rem .7rem;
    font-size: .72rem; font-weight: 600; color: var(--accent);
}
.topnav-badge .pulse {
    width: 6px; height: 6px; background: var(--accent);
    border-radius: 50%; animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%,100% { opacity: 1; transform: scale(1); }
    50%     { opacity: .4; transform: scale(.65); }
}

/* ── Hero ── */
.hero {
    position: relative; padding: 2.8rem 3rem 2.4rem;
    background:
        radial-gradient(ellipse 50% 70% at 80% 30%, rgba(62,207,218,.06), transparent 70%),
        radial-gradient(ellipse 40% 60% at 20% 80%, rgba(245,166,35,.03), transparent 60%),
        linear-gradient(160deg, var(--bg-card) 0%, var(--bg-surface) 100%);
    border: 1px solid var(--border); border-radius: var(--radius-xl);
    margin-bottom: 1.8rem; overflow: hidden;
}
.hero::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent 5%, var(--accent) 50%, transparent 95%); opacity: .6;
}
.hero-label {
    font-family: 'Space Grotesk', sans-serif; font-size: .7rem; font-weight: 600;
    letter-spacing: .14em; text-transform: uppercase; color: var(--accent); margin-bottom: .7rem;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif; font-size: clamp(2rem, 4.5vw, 3.4rem);
    font-weight: 700; letter-spacing: -.035em; line-height: 1.08;
    color: var(--ink) !important; margin: 0 0 .75rem;
}
.hero h1 em {
    font-style: normal;
    background: linear-gradient(135deg, var(--accent), #7ee8ef, var(--accent));
    background-size: 200% 200%;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 4s ease-in-out infinite;
}
@keyframes shimmer {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.hero-desc { color: var(--ink-secondary); font-size: 1.05rem; line-height: 1.65; max-width: 58ch; margin: 0 0 1.6rem; }
.hero-features { display: flex; gap: .55rem; flex-wrap: wrap; }
.hero-feat {
    display: inline-flex; align-items: center; gap: .4rem;
    background: rgba(255,255,255,.03); border: 1px solid var(--border);
    border-radius: 99px; padding: .32rem .85rem; font-size: .78rem; color: var(--ink-muted);
    transition: border-color .2s, color .2s;
}
.hero-feat:hover { border-color: var(--border-accent); color: var(--ink-secondary); }
.hero-feat .fi { color: var(--accent); font-size: .88em; }

/* ── Steps ── */
.steps {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: .7rem; margin-bottom: 1.8rem;
}
.step-card {
    position: relative; background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-md); padding: 1.1rem 1.2rem 1.1rem 4rem;
    transition: border-color .2s, box-shadow .25s, transform .15s;
}
.step-card:hover { border-color: var(--border-accent); box-shadow: var(--shadow-glow); transform: translateY(-2px); }
.step-num {
    position: absolute; left: 1rem; top: 50%; transform: translateY(-50%);
    width: 2.2rem; height: 2.2rem;
    background: linear-gradient(135deg, var(--accent-deep), var(--accent));
    border-radius: 50%; display: flex; align-items: center; justify-content: center;
    font-family: 'Space Grotesk', sans-serif; font-size: .8rem; font-weight: 700; color: #04222A;
    box-shadow: 0 0 16px var(--accent-glow);
}
.step-title { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: .92rem; color: var(--ink); margin-bottom: .15rem; }
.step-desc { font-size: .78rem; color: var(--ink-muted); line-height: 1.45; }

/* ── Panel Headers ── */
.panel-hdr { display: flex; align-items: center; gap: .6rem; margin-bottom: .9rem; padding-bottom: .7rem; border-bottom: 1px solid var(--border); }
.panel-ico {
    width: 2.2rem; height: 2.2rem; background: var(--accent-soft);
    border: 1px solid var(--border-accent); border-radius: var(--radius-xs);
    display: flex; align-items: center; justify-content: center; font-size: 1rem;
}
.panel-ttl { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.05rem; color: var(--ink); margin: 0; }
.panel-sub { font-size: .76rem; color: var(--ink-muted); margin-top: .05rem; }

/* ── File Uploader ── */
[data-testid="stFileUploader"] section {
    background: var(--bg-card) !important; border: 2px dashed var(--border-accent) !important;
    border-radius: var(--radius-md) !important; padding: 2.2rem 1.5rem !important;
    transition: border-color .25s, background .25s, box-shadow .25s;
}
[data-testid="stFileUploader"] section:hover {
    border-color: var(--accent) !important; background: var(--accent-soft) !important;
    box-shadow: var(--shadow-glow) !important;
}
[data-testid="stFileUploader"] section > div,
[data-testid="stFileUploaderDropzoneInstructions"] * { color: var(--ink-muted) !important; }
[data-testid="stFileUploader"] button {
    background: var(--accent-soft) !important; color: var(--accent) !important;
    border: 1px solid var(--border-accent) !important; border-radius: var(--radius-xs) !important;
    font-weight: 600 !important;
}
[data-testid="stFileUploader"] button:hover { background: rgba(62,207,218,.18) !important; box-shadow: 0 0 12px var(--accent-glow) !important; }
.privacy-tag { display: inline-flex; align-items: center; gap: .35rem; font-size: .72rem; color: var(--ink-faint); margin-top: .55rem; }

/* ── Images ── */
[data-testid="stImage"] {
    background: #030710 !important; border: 1px solid var(--border) !important;
    padding: .45rem !important; border-radius: var(--radius-md) !important;
    box-shadow: inset 0 2px 24px rgba(0,0,0,.5), var(--shadow-sm) !important; overflow: hidden !important;
}
[data-testid="stImage"] img { border-radius: var(--radius-xs) !important; display: block; max-width: 100%; }
[data-testid="stImageCaption"] { color: var(--ink-muted) !important; font-size: .73rem !important; text-align: center !important; margin-top: .35rem !important; }

/* ── Verdicts ── */
.verdict { border-radius: var(--radius-md); padding: 1.1rem 1.3rem; margin-bottom: 1rem; display: flex; align-items: flex-start; gap: 1rem; }
.verdict.is-warn { background: linear-gradient(135deg, rgba(245,166,35,.09), rgba(245,80,50,.04)); border: 1px solid rgba(245,166,35,.30); }
.verdict.is-ok { background: linear-gradient(135deg, rgba(52,211,153,.08), rgba(52,211,153,.02)); border: 1px solid rgba(52,211,153,.24); }
.verdict-ico { font-size: 1.8rem; line-height: 1; flex-shrink: 0; }
.verdict-title { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.05rem; margin: 0 0 .3rem; color: var(--ink); }
.verdict.is-warn .verdict-title { color: var(--warn); }
.verdict.is-ok  .verdict-title { color: var(--ok); }
.verdict-msg { font-size: .84rem; color: var(--ink-muted); margin: 0; line-height: 1.55; }

/* ── Finding Cards ── */
.finding { background: var(--bg-surface); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: .9rem 1.1rem; margin-bottom: .5rem; transition: border-color .15s, transform .1s; }
.finding:hover { border-color: var(--border-hover); transform: translateY(-1px); }
.finding.hi  { border-left: 3px solid var(--warn); }
.finding.mid { border-left: 3px solid #c8963a; }
.finding.lo  { border-left: 3px solid var(--ink-faint); }
.f-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: .5rem; }
.f-name { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: .95rem; color: var(--ink); }
.f-meta { display: flex; align-items: center; gap: .5rem; }
.f-pct { font-variant-numeric: tabular-nums; font-weight: 700; font-size: 1.1rem; color: var(--ink); }
.f-tag { font-size: .7rem; font-weight: 600; padding: .18rem .6rem; border-radius: 99px; }
.f-tag.hi  { background: rgba(245,166,35,.14); color: var(--warn); border: 1px solid rgba(245,166,35,.28); }
.f-tag.mid { background: rgba(200,150,58,.10); color: #e0a040; border: 1px solid rgba(200,150,58,.22); }
.f-tag.lo  { background: rgba(91,112,137,.10); color: var(--ink-muted); border: 1px solid var(--border); }
.cbar { height: 4px; background: var(--bg-base); border-radius: 99px; overflow: hidden; }
.cfill { height: 100%; border-radius: 99px; transition: width .5s ease; }
.cfill.hi  { background: linear-gradient(90deg, var(--warn), #f7c86e); }
.cfill.mid { background: linear-gradient(90deg, #c8963a, #e0b040); }
.cfill.lo  { background: linear-gradient(90deg, var(--ink-faint), #5b7089); }

/* ── Empty State ── */
.empty-box { background: var(--bg-card); border: 1px dashed rgba(62,207,218,.18); border-radius: var(--radius-lg); padding: 3.5rem 2rem; text-align: center; }
.empty-anim {
    width: 80px; height: 80px; margin: 0 auto 1.2rem; border-radius: 50%;
    background: radial-gradient(circle, var(--accent-soft), transparent 70%);
    display: flex; align-items: center; justify-content: center; font-size: 2.5rem;
    animation: float-icon 3s ease-in-out infinite;
}
@keyframes float-icon { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }
.empty-box h4 { font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; font-weight: 600; color: var(--ink); margin: 0 0 .4rem; }
.empty-box p { font-size: .88rem; color: var(--ink-muted); margin: 0; line-height: 1.55; max-width: 38ch; margin-inline: auto; }

/* ── Info Card ── */
.info-card { background: linear-gradient(135deg, rgba(62,207,218,.05), rgba(62,207,218,.01)); border: 1px solid var(--border-accent); border-radius: var(--radius-md); padding: 1rem 1.2rem; margin-top: 1.1rem; display: flex; gap: .8rem; align-items: flex-start; }
.info-card .ic-icon { font-size: 1.3rem; flex-shrink: 0; margin-top: .1rem; }
.info-card .ic-title { font-family: 'Space Grotesk', sans-serif; font-size: .88rem; font-weight: 600; color: var(--accent); margin-bottom: .2rem; }
.info-card p { font-size: .82rem; color: var(--ink-muted); margin: 0; line-height: 1.55; }

/* ── Widgets ── */
[data-testid="stExpander"] { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: var(--ink) !important; font-weight: 500 !important; }
[data-testid="stAlert"] { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-xs) !important; }
[data-testid="stAlert"] * { color: var(--ink) !important; }
[data-testid="stSpinner"] * { color: var(--accent) !important; }
[data-testid="stSlider"] [role="slider"] { background: var(--accent) !important; }
[data-testid="stSlider"] [data-baseweb="slider"] > div > div { background: var(--border) !important; }
[data-testid="stSlider"] [data-testid="stThumbValue"] { color: var(--accent) !important; }

/* ── Buttons ── */
.stButton > button, .stDownloadButton > button {
    border-radius: var(--radius-xs) !important; font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important; font-size: .85rem !important; padding: .6rem 1rem !important;
    transition: all .18s ease !important; border: 1px solid var(--border) !important;
    background: var(--bg-card) !important; color: var(--ink) !important;
}
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--accent) !important; color: var(--accent) !important; box-shadow: 0 0 16px var(--accent-glow) !important; transform: translateY(-1px) !important; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, var(--accent), var(--accent-deep)) !important; border: none !important; color: #04222A !important; }
.stButton > button[kind="primary"]:hover { box-shadow: 0 0 28px var(--accent-glow) !important; }
.stButton > button[kind="primary"] * { color: #04222A !important; }
.stDownloadButton > button { background: var(--accent-soft) !important; border: 1px solid var(--border-accent) !important; color: var(--accent) !important; }
.stDownloadButton > button:hover { background: rgba(62,207,218,.15) !important; box-shadow: 0 0 16px var(--accent-glow) !important; }

/* ── Typography ── */
h1,h2,h3,h4 { color: var(--ink) !important; }
hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }
p { line-height: 1.6; }
[data-testid="stMarkdownContainer"] * { color: var(--ink); }

/* ── Footer ── */
.app-footer { text-align: center; padding: 2rem 1rem 1rem; border-top: 1px solid var(--border); margin-top: 2.5rem; color: var(--ink-faint); font-size: .75rem; }

/* ── Responsive ── */
@media (max-width: 780px) {
    .block-container { padding: 1rem 1rem 3rem; }
    .hero { padding: 1.8rem 1.4rem 1.6rem; border-radius: var(--radius-lg); }
    .hero h1 { font-size: 1.8rem !important; }
    .steps { grid-template-columns: 1fr; }
    .topnav-right { gap: .7rem; }
    .topnav-link { display: none; }
}
@media (prefers-reduced-motion: reduce) { *, ::before, ::after { animation: none !important; transition: none !important; } }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────
VERIFIED_CLASSES = (
    "elbow_fracture",
    "finger_fracture",
    "forearm_fracture",
    "humerus_fracture",
    "shoulder_fracture",
    "wrist_fracture",
)
VERIFIED_CLASS_SCHEMAS = (VERIFIED_CLASSES, ("fracture",))


@st.cache_resource
def load_model():
    # Do not fall back to the old nine-class release: it was trained with
    # unverified FracAtlas anatomy labels and is the source of the shoulder bias.
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
    return YOLO(model_path)


model = load_model()


def display_name(name: str) -> str:
    """Turn the class name stored in a YOLO checkpoint into UI text."""
    return str(name).replace("_", " ").strip().capitalize()


# The checkpoint is the source of truth for class IDs. Keeping a second,
# hard-coded mapping here can silently display the wrong anatomy after a model
# is retrained with a corrected class list.
CLASS_NAMES = {
    int(class_id): display_name(name)
    for class_id, name in model.names.items()
}

MODEL_CLASSES = tuple(str(model.names[i]) for i in sorted(model.names))
if MODEL_CLASSES not in VERIFIED_CLASS_SCHEMAS:
    st.error(
        "This app has stopped an unsupported checkpoint. Deploy the verified fracture "
        "checkpoint and set FRACTURE_MODEL_PATH to its path before running the app."
    )
    st.stop()

IS_BINARY_FRACTURE_MODEL = MODEL_CLASSES == ("fracture",)


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
    # The lightweight retrained detector produces lower confidence scores than
    # the old model. A 45% default hid genuine detections, so use a lower
    # screening threshold and leave the stricter settings available.
    "Strict: fewer alerts": 20,
    "Balanced: fewer low-confidence alerts": 10,
    "Sensitive (recommended for screening)": 5,
    "Very sensitive: test low-confidence findings": 1,
}
MAX_MB = 10

# ─────────────────────────────────────────────────────────────
# Top navigation bar
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="topnav">
  <div class="topnav-left">
    <div class="topnav-logo">🩻</div>
    <div class="topnav-brand">Osteo<span>Scan</span> AI</div>
  </div>
  <div class="topnav-right">
    <span class="topnav-link">Fracture Screening</span>
    <span class="topnav-link">YOLO v8 Model</span>
    <div class="topnav-badge"><div class="pulse"></div>Online</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
  <div class="hero-label">Deep Learning · Computer Vision · Medical Imaging</div>
  <h1>Bone Fracture<br><em>Detection System</em></h1>
  <p class="hero-desc">
    Upload an X-ray and our deep-learning model instantly highlights
    suspected fracture sites — helping you prepare for a clinical consultation.
  </p>
  <div class="hero-features">
    <div class="hero-feat"><span class="fi">⚡</span>Real-time inference</div>
    <div class="hero-feat"><span class="fi">🔒</span>Image never saved</div>
    <div class="hero-feat"><span class="fi">📱</span>Works on any device</div>
    <div class="hero-feat"><span class="fi">🩻</span>YOLO v8 backbone</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# Workflow steps
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="steps">
  <div class="step-card">
    <div class="step-num">1</div>
    <div class="step-title">Upload</div>
    <div class="step-desc">Choose a clear JPG or PNG X-ray from your device.</div>
  </div>
  <div class="step-card">
    <div class="step-num">2</div>
    <div class="step-title">Auto-analyse</div>
    <div class="step-desc">The model runs instantly — no button needed.</div>
  </div>
  <div class="step-card">
    <div class="step-num">3</div>
    <div class="step-title">Review &amp; Share</div>
    <div class="step-desc">See annotated results and download for your doctor.</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# Main layout — Upload | Results
# ─────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown(
        """
<div class="panel-hdr">
  <div class="panel-ico">📤</div>
  <div>
    <div class="panel-ttl">Your X-ray</div>
    <div class="panel-sub">JPG or PNG · max 10 MB · image never stored</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload an X-ray (JPG or PNG, up to 10 MB)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        help="JPG or PNG, up to 10 MB.",
    )
    with st.expander("💡 Tips for a better result"):
        st.markdown(
            "- Use a clear X-ray, not a blurry or cropped photo.\n"
            "- Make sure the whole bone and the joints beside it are in the picture.\n"
            "- If you photograph a film, avoid glare and shadows.\n"
            "- Upload one body part at a time."
        )
    with st.expander("⚙️ Detection sensitivity"):
        level = st.select_slider(
            "How cautious should the check be?",
            options=list(SENSITIVITY.keys()),
            value="Sensitive (recommended for screening)",
            help="Sensitive shows more possible fractures but also more false alarms. "
            "Strict shows fewer, but may miss subtle ones.",
        )
    confidence = SENSITIVITY[level]
    st.markdown(
        '<div class="privacy-tag">🔒 This app does not save your image.</div>',
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
                    # Match the corrected model's 640px training resolution. The
                    # threshold comes from the selected screening sensitivity.
                    results = model.predict(
                        np.array(image),
                        imgsz=640,
                        conf=confidence / 100,
                        iou=0.45,
                        max_det=3,
                        verbose=False,
                    )
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
    st.markdown(
        """
<div class="panel-hdr">
  <div class="panel-ico">🔬</div>
  <div>
    <div class="panel-ttl">Analysis Results</div>
    <div class="panel-sub">AI-generated — always verify with a clinician</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    result = st.session_state.get("result")

    if analysis_error:
        st.error(analysis_error)
    elif image is None or not result:
        st.markdown(
            """
<div class="empty-box">
  <div class="empty-anim">🩻</div>
  <h4>Results will appear here</h4>
  <p>Upload an X-ray on the left — analysis starts automatically, no button needed.</p>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        rows = result["rows"]
        n = len(rows)

        if n == 0:
            st.markdown(
                """
<div class="verdict is-ok">
  <div class="verdict-ico">✅</div>
  <div>
    <p class="verdict-title">No High-Confidence Finding</p>
    <p class="verdict-msg">The model detected no strong fracture signal above the current threshold.
    This does <strong>not</strong> rule out a fracture. If you are in pain, please see a doctor.</p>
  </div>
</div>""",
                unsafe_allow_html=True,
            )
        else:
            regions = ", ".join(dict.fromkeys(r["Region"].replace(" fracture", "").lower() for r in rows))
            peak_confidence = rows[0]["Confidence"]
            finding_name = "a possible fracture" if IS_BINARY_FRACTURE_MODEL else regions
            if peak_confidence < 10:
                verdict_html = f"""
<div class="verdict is-warn">
  <div class="verdict-ico">⚠️</div>
  <div>
    <p class="verdict-title">Low-Confidence Signal</p>
    <p class="verdict-msg">Possible area of concern near: <strong>{finding_name}</strong>.
    This is a screening hint, not a diagnosis. A clinician must review the X-ray.</p>
  </div>
</div>"""
            else:
                verdict_html = f"""
<div class="verdict is-warn">
  <div class="verdict-ico">🚨</div>
  <div>
    <p class="verdict-title">Possible Fracture Detected</p>
    <p class="verdict-msg">Suspected region: <strong>{finding_name}</strong>.
    Review the annotated image below and show it to your doctor.</p>
  </div>
</div>"""
            st.markdown(verdict_html, unsafe_allow_html=True)

        st.image(
            result["image"],
            caption="Highlighted regions indicate where the model suspects a fracture.",
            use_container_width=True,
        )

        if n:
            st.markdown(
                """
<div class="panel-hdr" style="margin-top:1.2rem;margin-bottom:.3rem;">
  <div class="panel-ico">📊</div>
  <div><div class="panel-ttl">Confidence Breakdown</div></div>
</div>""",
                unsafe_allow_html=True,
            )
            for r in rows:
                band, label = confidence_band(r["Confidence"])
                st.markdown(
                    f"""
<div class="finding {band}">
  <div class="f-top">
    <span class="f-name">{r['Region']}</span>
    <div class="f-meta">
      <span class="f-pct">{r['Confidence']:.0f}%</span>
      <span class="f-tag {band}">{label}</span>
    </div>
  </div>
  <div class="cbar"><div class="cfill {band}" style="width:{r['Confidence']:.0f}%"></div></div>
</div>
""",
                    unsafe_allow_html=True,
                )

        st.markdown(
            """
<div class="info-card">
  <div class="ic-icon">🏥</div>
  <div>
    <div class="ic-title">What to do next</div>
    <p>Show this result and your original X-ray to a doctor or radiologist.
    Do <strong>not</strong> use it alone to decide on any treatment.</p>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.write("")
        stem = os.path.splitext(result["file"])[0]
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "⬇️ Save Annotated Image",
                data=to_png_bytes(result["image"]),
                file_name=f"{stem}_marked.png",
                mime="image/png",
                use_container_width=True,
            )
        with d2:
            if n:
                df = pd.DataFrame(rows).round({"Confidence": 1})
                st.download_button(
                    "⬇️ Export Results (CSV)",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{stem}_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="app-footer">
  <strong>OsteoScan AI</strong> · Fracture Detection System · For educational &amp; demonstration purposes only<br>
  Built with Streamlit &amp; YOLOv8 · Image never stored
</div>
""",
    unsafe_allow_html=True,
)
