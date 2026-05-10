"""
app.py — Face Recognition System
- Live webcam capture via browser (mobile-compatible using st.camera_input)
- Register faces from live screenshots OR uploaded photos
- Attendance / recognition log with timestamps
- Export database & logs
- Mobile-first responsive UI
"""

import streamlit as st
import cv2
import numpy as np
import json
import os
import time
import base64
import shutil
from datetime import datetime
from io import BytesIO
from PIL import Image
import torch
from model import FaceEmbeddingModel, get_face_detector
from database import FaceDatabase
from recognition import FaceRecognizer

# ── Environment detection ─────────────────────────────────────────────────────
IS_COLAB = "COLAB_JUPYTER_TOKEN" in os.environ or os.path.exists("/content")

# On Colab, use the DB path set by the notebook launcher (persisted to Drive)
# Falls back to local embeddings.json when running normally
DB_PATH = os.environ.get("FACEID_DB_PATH", "embeddings.json")
DB_PERSIST_PATH = os.environ.get("FACEID_DB_PERSIST", "")   # Drive path, if set

def _persist_db():
    """Copy local DB → Google Drive if running on Colab with Drive mounted."""
    if DB_PERSIST_PATH and os.path.exists(DB_PATH):
        try:
            shutil.copy2(DB_PATH, DB_PERSIST_PATH)
        except Exception:
            pass

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face ID System",
    page_icon="🪪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS (mobile-first, clean dark theme) ───────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;700&display=swap');

:root {
  --bg:      #080C14;
  --surface: #0E1520;
  --card:    #131C2B;
  --border:  #1E2D42;
  --primary: #3B82F6;
  --green:   #10B981;
  --amber:   #F59E0B;
  --red:     #EF4444;
  --text:    #E2E8F0;
  --muted:   #64748B;
  --mono:    'DM Mono', monospace;
  --sans:    'DM Sans', sans-serif;
}

html, body, [class*="css"] { font-family: var(--sans); background: var(--bg); color: var(--text); }
.stApp { background: var(--bg); }

/* Hide default streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.block-container { padding: 1rem 1rem 2rem; max-width: 1100px; }

/* Header */
.app-header {
  display: flex; align-items: center; gap: 1rem;
  padding: 1.2rem 1.5rem;
  background: linear-gradient(120deg, #0E1520 0%, #101e35 100%);
  border: 1px solid var(--border);
  border-radius: 14px;
  margin-bottom: 1.5rem;
}
.app-header .icon { font-size: 2.2rem; }
.app-header h1 {
  font-family: var(--mono); font-size: 1.4rem; font-weight: 500;
  background: linear-gradient(90deg, #60A5FA, #34D399);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin: 0; letter-spacing: -0.02em;
}
.app-header p { margin: 0; font-size: 0.8rem; color: var(--muted); }

/* Metric strip */
.metrics-row { display: flex; gap: 0.75rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric {
  flex: 1; min-width: 100px;
  background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  padding: 0.9rem 1rem; text-align: center;
}
.metric .val { font-family: var(--mono); font-size: 1.6rem; color: var(--primary); font-weight: 500; }
.metric .lbl { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  background: var(--card); border-radius: 10px; gap: 0; padding: 4px;
  border: 1px solid var(--border); flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 7px; color: var(--muted); font-weight: 500;
  padding: 0.45rem 1rem; font-size: 0.85rem;
}
.stTabs [aria-selected="true"] { background: var(--primary) !important; color: #fff !important; }

/* Buttons */
.stButton > button {
  background: var(--primary); color: #fff; font-weight: 600;
  border: none; border-radius: 8px; padding: 0.5rem 1.25rem;
  font-family: var(--mono); font-size: 0.85rem; width: 100%;
  transition: opacity 0.15s;
}
.stButton > button:hover { opacity: 0.85; }
.stButton > button:disabled { opacity: 0.35; cursor: not-allowed; }

/* Badges */
.badge {
  display: inline-block; border-radius: 20px;
  padding: 0.18rem 0.7rem; font-size: 0.78rem; font-weight: 600; font-family: var(--mono);
}
.badge-ok  { background: rgba(16,185,129,0.12); color: #10B981; border: 1px solid rgba(16,185,129,0.25); }
.badge-unk { background: rgba(239,68,68,0.12);  color: #EF4444; border: 1px solid rgba(239,68,68,0.25); }
.badge-reg { background: rgba(59,130,246,0.12); color: #60A5FA; border: 1px solid rgba(59,130,246,0.25); }

/* Status pill */
.pill-live    { display:inline-block; background:rgba(16,185,129,0.15); color:#10B981; border:1px solid rgba(16,185,129,0.3); border-radius:20px; padding:0.25rem 1rem; font-weight:700; font-family:var(--mono); font-size:0.82rem; }
.pill-stopped { display:inline-block; background:rgba(100,116,139,0.15); color:var(--muted); border:1px solid rgba(100,116,139,0.3); border-radius:20px; padding:0.25rem 1rem; font-weight:700; font-family:var(--mono); font-size:0.82rem; }

/* Log table */
.log-row {
  display: grid; grid-template-columns: 1fr 2fr 1fr 1fr;
  padding: 0.6rem 1rem; border-bottom: 1px solid var(--border);
  font-size: 0.82rem; align-items: center;
}
.log-row:first-child { background: var(--card); font-weight: 600; color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; }
.log-container { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }

/* Cards */
.info-card {
  background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  padding: 1rem 1.25rem; margin-bottom: 0.75rem;
}

/* Input fields */
.stTextInput input, .stSelectbox select {
  background: var(--card) !important; border: 1px solid var(--border) !important;
  color: var(--text) !important; border-radius: 8px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
  background: var(--card); border: 2px dashed var(--border); border-radius: 10px; padding: 0.5rem;
}

/* Sidebar */
[data-testid="stSidebar"] { background: var(--card) !important; border-right: 1px solid var(--border); }

/* Camera */
[data-testid="stCameraInput"] video, [data-testid="stCameraInput"] img {
  border-radius: 10px !important;
}

/* Slider */
.stSlider [data-testid="stThumbValue"] { color: var(--primary) !important; }

/* Responsive */
@media (max-width: 600px) {
  .app-header h1 { font-size: 1.1rem; }
  .metric .val { font-size: 1.3rem; }
  .log-row { grid-template-columns: 1fr 1.5fr 1fr; font-size: 0.75rem; }
  .log-row > *:last-child { display: none; }
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "db":            None,
    "recognizer":    None,
    "stats":         {"recognized": 0, "unknown": 0, "frames": 0},
    "attendance":    [],          # list of {time, name, confidence, source}
    "fps":           0.0,
    "video_running": False,
    "last_frame":    None,        # latest BGR frame for screenshot-register
    "last_results":  [],
    "register_snap": None,        # PIL image queued for registration
    "threshold":     0.55,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Lazy init heavy objects
if st.session_state.db is None:
    st.session_state.db = FaceDatabase(DB_PATH)
if st.session_state.recognizer is None:
    st.session_state.recognizer = FaceRecognizer(st.session_state.db)

db:         FaceDatabase  = st.session_state.db
recognizer: FaceRecognizer = st.session_state.recognizer

# ── Helper: log attendance ────────────────────────────────────────────────────
def log_attendance(name: str, confidence: float, source: str = "live"):
    # Deduplicate: skip if same name logged within last 30 seconds
    now = datetime.now()
    for entry in reversed(st.session_state.attendance[-20:]):
        if entry["name"] == name:
            last = datetime.fromisoformat(entry["time"])
            if (now - last).seconds < 30:
                return
    st.session_state.attendance.append({
        "time":       now.isoformat(timespec="seconds"),
        "name":       name,
        "confidence": confidence,
        "source":     source,
    })
    _persist_db()   # keep Drive in sync whenever a new recognition is logged

# ── Helper: pil → bgr ────────────────────────────────────────────────────────
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

# ── Helper: image download button ────────────────────────────────────────────
def download_image_btn(img_bgr: np.ndarray, filename: str, label: str):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    st.download_button(label, buf.getvalue(), filename, "image/png")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="icon">🪪</div>
  <div>
    <h1>FACE ID SYSTEM</h1>
    <p>Real-time recognition · FaceNet embeddings · Attendance tracking · Mobile-ready</p>
  </div>
</div>
""", unsafe_allow_html=True)

if IS_COLAB:
    drive_ok = bool(DB_PERSIST_PATH)
    drive_msg = "✅ Google Drive connected — database persists across sessions." if drive_ok \
                else "⚠️ Google Drive not mounted. Run Cell 3 in the notebook to persist your database."
    st.info(f"🟡 **Running on Google Colab** — browser camera mode active. {drive_msg}", icon="🔬")

# ── Metrics strip ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="metrics-row">
  <div class="metric"><div class="val">{len(db.get_all_names())}</div><div class="lbl">Registered</div></div>
  <div class="metric"><div class="val">{st.session_state.stats['recognized']}</div><div class="lbl">Recognized</div></div>
  <div class="metric"><div class="val">{st.session_state.stats['unknown']}</div><div class="lbl">Unknown</div></div>
  <div class="metric"><div class="val">{len(st.session_state.attendance)}</div><div class="lbl">Log Entries</div></div>
  <div class="metric"><div class="val">{st.session_state.fps:.1f}</div><div class="lbl">FPS</div></div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    threshold = st.slider("Confidence threshold", 0.3, 0.9,
                          st.session_state.threshold, 0.01,
                          help="Minimum cosine similarity to recognise a face")
    st.session_state.threshold = threshold
    recognizer.threshold       = threshold

    skip_frames = st.slider("Process every N frames", 1, 6, 2,
                            help="Higher = faster; lower = more accurate")

    st.markdown("---")
    st.markdown("### 🔧 Model")
    model_mode = getattr(get_face_detector(), "mode", "unknown")
    if model_mode == "facenet":
        st.success("FaceNet (deep learning)", icon="🧠")
    else:
        st.warning("HOG fallback (no facenet-pytorch)", icon="⚠️")

    st.markdown("---")
    st.markdown("### 📤 Export")
    if st.button("Export DB (JSON)"):
        st.download_button("⬇️ Download",
                           json.dumps(db.data, indent=2),
                           "face_db.json", "application/json")
    if st.session_state.attendance:
        log_csv = "time,name,confidence,source\n" + "\n".join(
            f"{e['time']},{e['name']},{e['confidence']:.3f},{e['source']}"
            for e in st.session_state.attendance
        )
        st.download_button("⬇️ Export Log (CSV)", log_csv, "attendance.csv", "text/csv")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_live, tab_snap, tab_reg, tab_db, tab_log = st.tabs([
    "🎥 Live",
    "📷 Snapshot",
    "➕ Register",
    "🗄️ Database",
    "📋 Log",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — LIVE VIDEO (continuous camera_input loop)
# ═══════════════════════════════════════════════════════════════
with tab_live:
    st.markdown("""
    <div class="info-card">
      <b>📱 Works on mobile & desktop.</b>  
      Allow camera access → frames are processed in real-time.  
      Hit <b>📸 Screenshot → Register</b> to save a face directly from the live feed.
    </div>
    """, unsafe_allow_html=True)

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
    with col_ctrl1:
        start_btn = st.button("▶ Start Camera",
                              disabled=st.session_state.video_running,
                              key="start_live")
    with col_ctrl2:
        stop_btn  = st.button("⏹ Stop Camera",
                              disabled=not st.session_state.video_running,
                              key="stop_live")

    if start_btn:
        st.session_state.video_running = True
        st.rerun()
    if stop_btn:
        st.session_state.video_running = False
        st.rerun()

    status_ph  = st.empty()
    frame_ph   = st.empty()
    result_ph  = st.empty()
    snap_ph    = st.empty()

    if st.session_state.video_running:
        status_ph.markdown('<span class="pill-live">⬤ LIVE</span>', unsafe_allow_html=True)

        # ── On Colab (or any remote server), skip cv2.VideoCapture — it has no
        #    physical camera. Use st.camera_input which streams from the browser. ──
        use_browser_cam = IS_COLAB

        if not use_browser_cam:
            # Try server-side OpenCV (works on localhost / machines with a webcam)
            cap = cv2.VideoCapture(0)
            use_browser_cam = not cap.isOpened()
            if use_browser_cam:
                cap.release()

        if not use_browser_cam:
            # ── Server-side OpenCV loop ───────────────────────────────────────
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            frame_idx    = 0
            last_results = []
            fps_t        = time.time()
            fps_count    = 0

            while st.session_state.video_running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Camera feed lost.")
                    break

                frame_idx  += 1
                fps_count  += 1
                st.session_state.last_frame = frame.copy()

                if frame_idx % skip_frames == 0:
                    last_results, annotated = recognizer.recognize_image(frame)
                    st.session_state.last_results = last_results
                    for r in last_results:
                        st.session_state.stats["frames"] += 1
                        if r["recognized"]:
                            st.session_state.stats["recognized"] += 1
                            log_attendance(r["name"], r["confidence"], "live")
                        else:
                            st.session_state.stats["unknown"] += 1
                else:
                    annotated = frame.copy()
                    for r in last_results:
                        recognizer.draw_result(annotated, r)

                # FPS overlay
                elapsed = time.time() - fps_t
                if elapsed >= 1.0:
                    st.session_state.fps = fps_count / elapsed
                    fps_count = 0
                    fps_t     = time.time()

                cv2.putText(annotated,
                            f"FPS:{st.session_state.fps:.1f}  Faces:{len(last_results)}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (59, 130, 246), 2, cv2.LINE_AA)

                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_ph.image(rgb, channels="RGB", use_column_width=True)

                if last_results:
                    html = f"<div style='margin-top:0.5rem'><b>👥 {len(last_results)} face(s)</b>&nbsp;&nbsp;"
                    for r in last_results:
                        if r["recognized"]:
                            html += f'<span class="badge badge-ok">✓ {r["name"]} {r["confidence"]:.0%}</span> '
                        else:
                            html += '<span class="badge badge-unk">? Unknown</span> '
                    html += "</div>"
                    result_ph.markdown(html, unsafe_allow_html=True)

                time.sleep(0.01)

            cap.release()
            status_ph.markdown('<span class="pill-stopped">⏹ Stopped</span>', unsafe_allow_html=True)

        else:
            # ── Browser camera (Colab / mobile / remote servers) ─────────────
            # st.camera_input streams directly from the user's browser camera.
            # Each button press = one captured frame → processed immediately.
            source_label = "Colab/Remote" if IS_COLAB else "Remote server"
            st.info(
                f"📡 **{source_label} mode** — camera feed comes from your browser. "
                "Each capture is processed instantly. Click repeatedly for continuous recognition.",
                icon="📱"
            )

            camera_frame = st.camera_input("Live Camera Feed", label_visibility="collapsed",
                                           key="live_cam_input")
            if camera_frame:
                t0  = time.time()
                pil = Image.open(camera_frame).convert("RGB")
                frame = pil_to_bgr(pil)
                st.session_state.last_frame = frame.copy()

                results, annotated = recognizer.recognize_image(frame)
                st.session_state.last_results = results

                elapsed = time.time() - t0
                st.session_state.fps = round(1.0 / max(elapsed, 0.001), 1)

                for r in results:
                    st.session_state.stats["frames"] += 1
                    if r["recognized"]:
                        st.session_state.stats["recognized"] += 1
                        log_attendance(r["name"], r["confidence"], "browser")
                    else:
                        st.session_state.stats["unknown"] += 1

                # FPS overlay on annotated frame
                cv2.putText(annotated,
                            f"Faces:{len(results)}  {elapsed*1000:.0f}ms",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (59, 130, 246), 2, cv2.LINE_AA)

                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_ph.image(rgb, channels="RGB", use_column_width=True, caption="Processed frame")

                if results:
                    html = "<div style='margin-top:0.5rem'>"
                    for r in results:
                        if r["recognized"]:
                            html += f'<span class="badge badge-ok">✓ {r["name"]} {r["confidence"]:.0%}</span> '
                        else:
                            html += '<span class="badge badge-unk">? Unknown</span> '
                    html += "</div>"
                    result_ph.markdown(html, unsafe_allow_html=True)

    else:
        status_ph.markdown('<span class="pill-stopped">⏹ Stopped</span>', unsafe_allow_html=True)
        frame_ph.markdown("""
        <div style="background:var(--card);border:1px solid var(--border);border-radius:12px;
                    padding:4rem 2rem;text-align:center;color:var(--muted);">
          <div style="font-size:3rem;">📷</div>
          <div style="margin-top:0.75rem;">Press <b>▶ Start Camera</b> to begin</div>
        </div>""", unsafe_allow_html=True)

    # ── Screenshot → Register shortcut ───────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📸 Screenshot → Register")
    st.caption("Capture the current frame and register a face directly from it.")

    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        if st.button("📸 Capture Frame", disabled=st.session_state.last_frame is None):
            frame = st.session_state.last_frame
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.session_state.register_snap = Image.fromarray(rgb)
            st.success("Frame captured! Go to **➕ Register** tab to save.")

    if st.session_state.register_snap:
        with col_s2:
            st.image(st.session_state.register_snap, width=220, caption="Captured frame")

# ═══════════════════════════════════════════════════════════════
# TAB 2 — SNAPSHOT (single image, upload or camera)
# ═══════════════════════════════════════════════════════════════
with tab_snap:
    st.markdown("#### Recognize faces in a photo")

    method = st.radio("Source", ["📁 Upload", "📷 Take Photo"],
                      horizontal=True, label_visibility="collapsed")
    image_input = None

    if method == "📁 Upload":
        up = st.file_uploader("Image file", type=["jpg","jpeg","png","webp"],
                              label_visibility="collapsed")
        if up:
            image_input = Image.open(up).convert("RGB")
    else:
        cam = st.camera_input("Take photo", label_visibility="collapsed")
        if cam:
            image_input = Image.open(cam).convert("RGB")
            st.session_state.last_frame = pil_to_bgr(image_input)

    if image_input:
        img_bgr = pil_to_bgr(image_input)
        with st.spinner("Detecting & recognising…"):
            results, annotated = recognizer.recognize_image(img_bgr)

        c_img, c_res = st.columns([3, 2])
        with c_img:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     use_column_width=True, caption="Result")
            download_image_btn(annotated, "result.png", "⬇️ Save annotated image")
        with c_res:
            st.markdown(f"**{len(results)} face(s) detected**")
            for i, r in enumerate(results):
                if r["recognized"]:
                    st.markdown(f'**Face #{i+1}** <span class="badge badge-ok">✓ {r["name"]}</span>  `{r["confidence"]:.1%}`',
                                unsafe_allow_html=True)
                    log_attendance(r["name"], r["confidence"], "snapshot")
                    st.session_state.stats["recognized"] += 1
                else:
                    st.markdown(f'**Face #{i+1}** <span class="badge badge-unk">? Unknown</span>',
                                unsafe_allow_html=True)
                    st.session_state.stats["unknown"] += 1
                st.divider()

            if any(not r["recognized"] for r in results):
                st.info("Unknown faces found — capture and register them in **➕ Register**.")
                if st.button("📸 Use this photo for registration"):
                    st.session_state.register_snap = image_input
                    st.success("Queued! Switch to ➕ Register tab.")

# ═══════════════════════════════════════════════════════════════
# TAB 3 — REGISTER
# ═══════════════════════════════════════════════════════════════
with tab_reg:
    st.markdown("#### Register a new person")

    # ── Source selector ───────────────────────────────────────────────────────
    reg_source = st.radio(
        "Registration source",
        ["📁 Upload photos", "📷 Take photo now", "🎥 Use captured frame"],
        horizontal=True, label_visibility="collapsed",
    )

    reg_images_pil = []   # list of PIL images to embed

    if reg_source == "📁 Upload photos":
        uploaded = st.file_uploader(
            "Upload 1–10 face photos (more = better accuracy)",
            type=["jpg","jpeg","png","webp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded:
            reg_images_pil = [Image.open(f).convert("RGB") for f in uploaded]
            cols = st.columns(min(len(reg_images_pil), 4))
            for idx, img in enumerate(reg_images_pil[:4]):
                with cols[idx]: st.image(img, use_column_width=True)

    elif reg_source == "📷 Take photo now":
        cam_reg = st.camera_input("Camera", label_visibility="collapsed", key="reg_cam")
        if cam_reg:
            pil = Image.open(cam_reg).convert("RGB")
            reg_images_pil = [pil]
            st.image(pil, width=300, caption="Photo for registration")
            st.session_state.last_frame = pil_to_bgr(pil)

    else:  # captured frame
        if st.session_state.register_snap:
            reg_images_pil = [st.session_state.register_snap]
            st.image(st.session_state.register_snap, width=300, caption="Captured frame")
            if st.button("🗑️ Clear captured frame"):
                st.session_state.register_snap = None
                st.rerun()
        else:
            st.warning("No frame captured yet. Use the 📸 Capture Frame button in the 🎥 Live tab first.")

    # ── Name input & submit ───────────────────────────────────────────────────
    person_name = st.text_input("Full name", placeholder="e.g. Ahmed Hassan",
                                key="reg_name")

    if st.button("✅ Register Person",
                 disabled=not (person_name.strip() and reg_images_pil)):
        prog = st.progress(0, text="Extracting face embeddings…")
        all_embs = []
        face_counts = []

        for i, pil_img in enumerate(reg_images_pil):
            bgr  = pil_to_bgr(pil_img)
            embs = recognizer.extract_embeddings(bgr)
            face_counts.append(len(embs))
            all_embs.extend(embs)
            prog.progress((i + 1) / len(reg_images_pil),
                          text=f"Image {i+1}/{len(reg_images_pil)} — {len(embs)} face(s)")

        if all_embs:
            avg_emb = np.mean(all_embs, axis=0)
            db.add_person(person_name.strip(), avg_emb.tolist())
            _persist_db()
            st.session_state.register_snap = None
            total_faces = sum(face_counts)
            st.success(f"✅ **{person_name.strip()}** registered using {total_faces} face embedding(s) from {len(reg_images_pil)} photo(s).")
            st.balloons()
        else:
            st.error("❌ No faces detected in the provided image(s). Use a clear, front-facing photo.")

# ═══════════════════════════════════════════════════════════════
# TAB 4 — DATABASE
# ═══════════════════════════════════════════════════════════════
with tab_db:
    st.markdown("#### Registered persons")

    names = db.get_all_names()
    col_search, col_count = st.columns([3, 1])
    with col_search:
        search = st.text_input("Search", placeholder="Filter by name…",
                               label_visibility="collapsed")
    with col_count:
        st.markdown(f'<div class="metric"><div class="val">{len(names)}</div><div class="lbl">Total</div></div>',
                    unsafe_allow_html=True)

    filtered = [n for n in names if search.lower() in n.lower()] if search else names

    if not filtered:
        st.info("No persons registered yet. Go to **➕ Register** to add faces.")
    else:
        for name in filtered:
            c1, c2, c3, c4 = st.columns([4, 2, 1, 1])
            with c1:
                st.markdown(f"👤 **{name}**")
            with c2:
                st.markdown('<span class="badge badge-reg">Registered</span>', unsafe_allow_html=True)
            with c3:
                # How many times recognized in log
                count = sum(1 for e in st.session_state.attendance if e["name"] == name)
                st.markdown(f"<code>{count} seen</code>", unsafe_allow_html=True)
            with c4:
                if st.button("🗑️", key=f"del_{name}", help=f"Remove {name}"):
                    db.remove_person(name)
                    st.rerun()

    st.markdown("---")
    da1, da2 = st.columns(2)
    with da1:
        st.download_button(
            "⬇️ Export database (JSON)",
            json.dumps(db.data, indent=2),
            "face_db.json", "application/json",
        )
    with da2:
        if st.button("🗑️ Clear all persons", type="secondary"):
            db.clear()
            st.warning("Database cleared.")
            st.rerun()

    # Import database
    st.markdown("---")
    st.markdown("#### 📥 Import database")
    imp_file = st.file_uploader("Import JSON", type=["json"], label_visibility="collapsed",
                                key="import_db")
    if imp_file:
        try:
            imported = json.load(imp_file)
            merged   = 0
            for k, v in imported.items():
                if k not in db.data:
                    db.add_person(k, v)
                    merged += 1
            st.success(f"Imported {merged} new person(s). ({len(imported) - merged} duplicates skipped)")
            st.rerun()
        except Exception as e:
            st.error(f"Import failed: {e}")

# ═══════════════════════════════════════════════════════════════
# TAB 5 — ATTENDANCE LOG
# ═══════════════════════════════════════════════════════════════
with tab_log:
    st.markdown("#### Attendance / Recognition Log")
    st.caption("Auto-logged when a registered face is recognised. Deduplication: 30-second cooldown per person.")

    log = st.session_state.attendance

    if not log:
        st.info("No recognitions logged yet. Start the live camera or run a snapshot.")
    else:
        # Summary cards
        unique_names = list({e["name"] for e in log})
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(f'<div class="metric"><div class="val">{len(log)}</div><div class="lbl">Total events</div></div>',
                        unsafe_allow_html=True)
        with s2:
            st.markdown(f'<div class="metric"><div class="val">{len(unique_names)}</div><div class="lbl">Unique persons</div></div>',
                        unsafe_allow_html=True)
        with s3:
            avg_conf = np.mean([e["confidence"] for e in log])
            st.markdown(f'<div class="metric"><div class="val">{avg_conf:.0%}</div><div class="lbl">Avg confidence</div></div>',
                        unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Log table
        html = """
        <div class="log-container">
          <div class="log-row">
            <span>Time</span><span>Name</span><span>Confidence</span><span>Source</span>
          </div>
        """
        for entry in reversed(log):
            t   = entry["time"].replace("T", " ")
            n   = entry["name"]
            c   = f"{entry['confidence']:.1%}"
            src = entry.get("source", "—")
            html += f'<div class="log-row"><span>{t}</span><span><b>{n}</b></span><span>{c}</span><span>{src}</span></div>'
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        log_csv = "time,name,confidence,source\n" + "\n".join(
            f"{e['time']},{e['name']},{e['confidence']:.4f},{e['source']}"
            for e in log
        )
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button("⬇️ Export log (CSV)", log_csv, "attendance.csv", "text/csv")
        with col_dl2:
            if st.button("🗑️ Clear log"):
                st.session_state.attendance = []
                st.session_state.stats = {"recognized": 0, "unknown": 0, "frames": 0}
                st.rerun()
