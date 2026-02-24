import streamlit as st
import requests
from PIL import Image, ImageDraw
import io
import base64
import os
import time

st.set_page_config(layout="wide", page_title="Locus Lens")

GATEWAY_URL = "http://localhost:8000"

# ─── Styling ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
    h1, h2, h3 { font-family: 'Syne', sans-serif; }
    .stApp { background: #0a0a0a; color: #f0ede8; }

    /* ── Shared ── */
    .step-badge {
        display: inline-block;
        background: #e8c547;
        color: #0d0d0d;
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 0.75rem;
        padding: 2px 10px;
        border-radius: 2px;
        margin-right: 8px;
        letter-spacing: 0.05em;
    }
    .store-tag {
        font-size: 0.65rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    div[data-testid="stButton"] button {
        background: #e8c547;
        color: #0d0d0d;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        border: none;
        letter-spacing: 0.05em;
    }
    div[data-testid="stButton"] button:hover { background: #f0d060; }

    /* ── Loading Screen Animations ── */
    @keyframes spin {
        from { transform: rotate(0deg); }
        to   { transform: rotate(360deg); }
    }
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 20px rgba(232,197,71,0.1); }
        50%       { box-shadow: 0 0 40px rgba(232,197,71,0.3); }
    }
    @keyframes shimmer {
        0%   { background-position: -600px 0; }
        100% { background-position: 600px 0; }
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse-badge {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.5; }
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0; }
    }
    @keyframes scan-line {
        0%   { top: 0%; opacity: 0.6; }
        100% { top: 100%; opacity: 0; }
    }

    /* ── Loading Wrapper ── */
    .loading-outer {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 60px 20px 40px;
        animation: fadeUp 0.7s ease both;
    }

    /* ── Spinner Ring ── */
    .spinner-wrap {
        position: relative;
        width: 100px;
        height: 100px;
        margin-bottom: 36px;
    }
    .spinner-ring {
        position: absolute;
        inset: 0;
        border-radius: 50%;
        border: 2px solid #1a1a1a;
        border-top-color: #e8c547;
        animation: spin 1s linear infinite, pulse-glow 2s ease-in-out infinite;
    }
    .spinner-ring-inner {
        position: absolute;
        inset: 12px;
        border-radius: 50%;
        border: 1px solid #161616;
        border-bottom-color: rgba(232,197,71,0.3);
        animation: spin 1.6s linear infinite reverse;
    }
    .spinner-dot {
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
    }

    /* ── Titles ── */
    .loading-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #f0ede8;
        letter-spacing: -0.02em;
        margin-bottom: 8px;
        text-align: center;
    }
    .loading-subtitle {
        font-size: 0.8rem;
        color: #444;
        text-align: center;
        max-width: 400px;
        line-height: 1.7;
        margin-bottom: 48px;
        font-family: 'DM Mono', monospace;
    }
    .loading-subtitle span {
        color: #e8c547;
        animation: blink 1.2s step-end infinite;
    }

    /* ── Service Cards ── */
    .services-grid {
        width: 100%;
        max-width: 500px;
        display: flex;
        flex-direction: column;
        gap: 8px;
        margin-bottom: 40px;
    }
    .service-card {
        background: #0f0f0f;
        border-radius: 10px;
        padding: 14px 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border: 1px solid #1a1a1a;
        transition: border-color 0.6s ease, background 0.6s ease;
        animation: fadeUp 0.5s ease both;
        overflow: hidden;
        position: relative;
    }
    .service-card::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 3px;
        border-radius: 3px 0 0 3px;
        background: #1a1a1a;
        transition: background 0.6s ease;
    }
    .service-card.ready {
        border-color: rgba(71,232,163,0.2);
        background: linear-gradient(135deg, #0a150f 0%, #0f0f0f 60%);
    }
    .service-card.ready::before  { background: #47e8a3; }
    .service-card.loading::before { background: #e8c547; }
    .service-card.error::before  { background: #e87447; }

    /* scan line effect on loading cards */
    .service-card.loading::after {
        content: '';
        position: absolute;
        left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(232,197,71,0.4), transparent);
        animation: scan-line 2s linear infinite;
    }

    .service-left {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .service-icon {
        font-size: 1.3rem;
        width: 28px;
        text-align: center;
        flex-shrink: 0;
    }
    .service-name {
        font-family: 'Syne', sans-serif;
        font-size: 0.78rem;
        font-weight: 700;
        color: #ccc;
        letter-spacing: 0.03em;
    }
    .service-desc {
        font-size: 0.6rem;
        color: #3a3a3a;
        font-family: 'DM Mono', monospace;
        margin-top: 2px;
    }

    /* ── Status Badges ── */
    .status-badge {
        font-size: 0.6rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        flex-shrink: 0;
        font-family: 'DM Mono', monospace;
    }
    .badge-ready   {
        background: rgba(71,232,163,0.08);
        color: #47e8a3;
        border: 1px solid rgba(71,232,163,0.25);
    }
    .badge-loading {
        background: rgba(232,197,71,0.08);
        color: #e8c547;
        border: 1px solid rgba(232,197,71,0.25);
        animation: pulse-badge 1.4s ease-in-out infinite;
    }
    .badge-error   {
        background: rgba(232,116,71,0.08);
        color: #e87447;
        border: 1px solid rgba(232,116,71,0.25);
    }

    /* ── Progress Bar ── */
    .progress-section {
        width: 100%;
        max-width: 500px;
        margin-bottom: 20px;
    }
    .progress-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.6rem;
        color: #333;
        font-family: 'DM Mono', monospace;
        margin-bottom: 8px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .progress-track {
        height: 2px;
        background: #161616;
        border-radius: 2px;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        border-radius: 2px;
        background: linear-gradient(90deg, #e8c547 0%, #f0d060 50%, #e8c547 100%);
        background-size: 600px 100%;
        animation: shimmer 1.8s linear infinite;
    }

    /* ── Hint ── */
    .refresh-hint {
        font-size: 0.62rem;
        color: #252525;
        font-family: 'DM Mono', monospace;
        letter-spacing: 0.06em;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("<h1 style='font-size:2.5rem; margin-bottom:0; font-family:Syne,sans-serif;'>🔎 LOCUS LENS</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#555; margin-top:4px; font-size:0.8rem; font-family:DM Mono,monospace;'>Upload a photo — AI detects every item — click what you want to find</p>", unsafe_allow_html=True)
st.divider()


# ─── Health Check ─────────────────────────────────────────────────────────────
def check_health():
    try:
        resp = requests.get(f"{GATEWAY_URL}/health", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("ready", False), data.get("services", {})
    except Exception:
        pass
    return False, {}


def render_loading_screen(services):
    ready_count  = sum(1 for v in services.values() if v == "ready")
    total_count  = max(len(services), 1)
    progress_pct = int((ready_count / total_count) * 100)

    SERVICE_META = {
        "gateway":       ("🌐", "Gateway",        "API routing & orchestration"),
        "visual_engine": ("🧠", "Vision Engine",   "CLIP · DeepFashion2 · YOLOv8"),
        "qdrant":        ("🗄️",  "Vector Database", "Qdrant similarity search"),
    }

    cards_html = ""
    for i, (key, meta) in enumerate(SERVICE_META.items()):
        icon, name, desc = meta
        status = services.get(key, "loading")

        if status == "ready":
            card_cls  = "ready"
            badge_cls = "badge-ready"
            badge_txt = "● &nbsp;READY"
        elif status == "loading":
            card_cls  = "loading"
            badge_cls = "badge-loading"
            badge_txt = "◌ &nbsp;LOADING"
        else:
            card_cls  = "error"
            badge_cls = "badge-error"
            badge_txt = "✕ &nbsp;ERROR"

        cards_html += f"""
        <div class="service-card {card_cls}" style="animation-delay:{i*0.1}s">
            <div class="service-left">
                <div class="service-icon">{icon}</div>
                <div>
                    <div class="service-name">{name}</div>
                    <div class="service-desc">{desc}</div>
                </div>
            </div>
            <div class="status-badge {badge_cls}">{badge_txt}</div>
        </div>
        """

    st.markdown(f"""
        <div class="loading-outer">

            <div class="spinner-wrap">
                <div class="spinner-ring"></div>
                <div class="spinner-ring-inner"></div>
                <div class="spinner-dot">🔎</div>
            </div>

            <div class="loading-title">Warming Up</div>
            <div class="loading-subtitle">
                AI models are loading into memory.<br>
                This takes 3–5 min on first launch<span>_</span>
            </div>

            <div class="services-grid">
                {cards_html}
            </div>

            <div class="progress-section">
                <div class="progress-label">
                    <span>Loading progress</span>
                    <span>{ready_count} / {total_count} services ready</span>
                </div>
                <div class="progress-track">
                    <div class="progress-fill" style="width:{progress_pct}%;"></div>
                </div>
            </div>

            <div class="refresh-hint">auto-refreshing every 5 seconds</div>

        </div>
    """, unsafe_allow_html=True)


# ─── Check if backend is ready ───────────────────────────────────────────────
is_ready, services = check_health()

if not is_ready:
    render_loading_screen(services)
    time.sleep(5)
    st.rerun()
    st.stop()

# ─── Backend is ready — show the main app ────────────────────────────────────

# ─── Session State ────────────────────────────────────────────────────────────
if "detections" not in st.session_state:
    st.session_state.detections = []
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = None
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "uploaded_bytes" not in st.session_state:
    st.session_state.uploaded_bytes = None

# ─── STEP 1: Upload ───────────────────────────────────────────────────────────
st.markdown("<span class='step-badge'>STEP 1</span> Upload your photo", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

if uploaded_file:
    new_bytes = uploaded_file.read()

    if new_bytes != st.session_state.uploaded_bytes:
        st.session_state.uploaded_bytes = new_bytes
        st.session_state.detections = []
        st.session_state.original_image = None
        st.session_state.selected_idx = None
        st.session_state.search_results = None

    # ─── STEP 2: Detect ──────────────────────────────────────────────────────
    if not st.session_state.detections:
        st.markdown("<span class='step-badge'>STEP 2</span> AI is scanning the image for items…", unsafe_allow_html=True)

        with st.spinner("🔍 Detecting fashion items..."):
            try:
                files = {"file": (uploaded_file.name, new_bytes, uploaded_file.type)}
                resp = requests.post(f"{GATEWAY_URL}/detect", files=files, timeout=120)

                if resp.status_code == 200:
                    result = resp.json()
                    st.session_state.detections = result.get("detections", [])
                    st.session_state.original_image = Image.open(io.BytesIO(new_bytes)).convert("RGB")

                    if not st.session_state.detections:
                        st.warning("No fashion items detected. Try a clearer photo with clothing visible.")
                else:
                    st.error(f"Detection failed: {resp.status_code}")
            except Exception as e:
                st.error(f"Connection error: {e}")

# ─── STEP 2 RESULTS ───────────────────────────────────────────────────────────
if st.session_state.detections and st.session_state.original_image:
    detections = st.session_state.detections
    orig_img   = st.session_state.original_image

    annotated = orig_img.copy()
    draw = ImageDraw.Draw(annotated)
    COLORS = ["#e8c547", "#47c5e8", "#e847a3", "#47e8a3", "#e87447", "#a347e8"]

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        color = COLORS[i % len(COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label_text = f"  {i+1}. {det['label'].upper()}  "
        draw.rectangle([x1, y1 - 22, x1 + len(label_text) * 7, y1], fill=color)
        draw.text((x1 + 4, y1 - 19), label_text.strip(), fill="#0d0d0d")

    col_img, col_select = st.columns([2, 1])

    with col_img:
        st.markdown(f"**{len(detections)} item{'s' if len(detections) > 1 else ''} detected** — select one on the right →")
        st.image(annotated, use_container_width=True)

    with col_select:
        st.markdown("<span class='step-badge'>STEP 2</span> Which item do you want to find?", unsafe_allow_html=True)

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            color = COLORS[i % len(COLORS)]

            patch = orig_img.crop((x1, y1, x2, y2))
            patch_buf = io.BytesIO()
            patch.save(patch_buf, format="PNG")
            patch_b64 = base64.b64encode(patch_buf.getvalue()).decode()

            is_selected   = st.session_state.selected_idx == i
            border_style  = f"3px solid {color}" if is_selected else "1px solid #1a1a1a"
            bg            = "#141414" if is_selected else "#0f0f0f"

            source = det.get("source", "")
            source_tag = "👗 DeepFashion2" if source == "deepfashion2" else ("👟 YOLO COCO" if source == "yolo_coco" else "🔍 CLIP")

            st.markdown(f"""
                <div style="border:{border_style}; border-radius:8px; overflow:hidden;
                            margin-bottom:10px; background:{bg}; transition:all 0.2s;">
                    <img src="data:image/png;base64,{patch_b64}"
                         style="width:100%; display:block; max-height:110px; object-fit:cover;">
                    <div style="padding:8px 12px;">
                        <div style="font-family:Syne,sans-serif; font-weight:700;
                                    font-size:0.78rem; color:{color}; letter-spacing:0.03em;">
                            {i+1}. {det['label'].upper()}
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-top:4px;">
                            <span style="font-size:0.6rem; color:#333; font-family:'DM Mono',monospace;">{source_tag}</span>
                            <span style="font-size:0.6rem; color:#333; font-family:'DM Mono',monospace;">{int(det['score']*100)}% conf</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            btn_label = "✓ Selected" if is_selected else "Select this item"
            if st.button(btn_label, key=f"select_{i}"):
                st.session_state.selected_idx = i
                st.session_state.search_results = None
                st.rerun()

    # ─── STEP 3: Search ──────────────────────────────────────────────────────
    if st.session_state.selected_idx is not None:
        st.divider()
        selected = detections[st.session_state.selected_idx]
        x1, y1, x2, y2 = selected["bbox"]

        st.markdown(f"<span class='step-badge'>STEP 3</span> Searching for <strong>{selected['label'].upper()}</strong> in the inventory…", unsafe_allow_html=True)

        col_srch, _ = st.columns([1, 3])
        with col_srch:
            search_btn = st.button("🔍 Find Similar Items", type="primary")

        if search_btn or (st.session_state.search_results is not None):

            if search_btn:
                with st.spinner("⚙️ Processing your selection..."):
                    try:
                        files = {"file": ("image.png", st.session_state.uploaded_bytes, "image/png")}
                        data  = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                        resp  = requests.post(f"{GATEWAY_URL}/search", files=files, data=data, timeout=60)
                        if resp.status_code == 200:
                            st.session_state.search_results = resp.json()
                        else:
                            st.error(f"Search failed: {resp.status_code} — {resp.text}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")

            if st.session_state.search_results:
                result_data      = st.session_state.search_results
                matches          = result_data.get("matches", [])
                debug_image_b64  = result_data.get("debug_image")
                detected_category = result_data.get("detected_category")

                st.divider()

                col_a, col_b, col_c = st.columns([1, 1, 3])

                with col_a:
                    st.markdown("**Your Selection**")
                    patch = st.session_state.original_image.crop((x1, y1, x2, y2))
                    st.image(patch, use_container_width=True)

                with col_b:
                    st.markdown("**AI Vision**")
                    if debug_image_b64:
                        ai_img = Image.open(io.BytesIO(base64.b64decode(debug_image_b64)))
                        st.image(ai_img, use_container_width=True)

                with col_c:
                    df2_label = selected['label'].upper()
                    if detected_category:
                        st.markdown(f"""
                            <div style="background:#0f0f0f; border:1px solid rgba(232,197,71,0.3);
                                        border-left: 3px solid #e8c547;
                                        border-radius:8px; padding:20px; margin-top:10px;">
                                <div style="font-family:'DM Mono',monospace; font-size:0.6rem;
                                            color:#555; letter-spacing:0.12em; text-transform:uppercase;">Detected Item</div>
                                <div style="font-family:Syne,sans-serif; font-size:1.7rem;
                                            font-weight:800; color:#e8c547; margin:6px 0 14px;">{df2_label}</div>
                                <div style="font-family:'DM Mono',monospace; font-size:0.6rem;
                                            color:#555; letter-spacing:0.12em; text-transform:uppercase;">Search Filter</div>
                                <div style="font-family:Syne,sans-serif; font-size:1.1rem;
                                            font-weight:700; color:#888; margin-top:4px;">{detected_category.upper()}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div style="background:#0f0f0f; border:1px solid #1a1a1a;
                                        border-left: 3px solid #333;
                                        border-radius:8px; padding:20px; margin-top:10px;">
                                <div style="font-family:'DM Mono',monospace; font-size:0.6rem;
                                            color:#555; letter-spacing:0.12em; text-transform:uppercase;">Detected Item</div>
                                <div style="font-family:Syne,sans-serif; font-size:1.7rem;
                                            font-weight:800; color:#555; margin:6px 0 8px;">{df2_label}</div>
                                <div style="font-size:0.72rem; color:#333; font-family:'DM Mono',monospace;">
                                    No category filter — showing all results
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                st.markdown(f"<h3 style='font-family:Syne,sans-serif; margin-top:28px; font-size:1.2rem;'>🎯 Top Matches &nbsp;<span style='color:#444; font-size:0.8rem; font-family:DM Mono,monospace;'>({len(matches)} found)</span></h3>", unsafe_allow_html=True)

                if matches:
                    cols = st.columns(5)
                    for idx, item in enumerate(matches):
                        with cols[idx % 5]:
                            local_path = os.path.join("demo_images", item['image_filename'])
                            if os.path.exists(local_path):
                                st.image(local_path, use_container_width=True)

                            medal = "🥇" if idx == 0 else ("🥈" if idx == 1 else ("🥉" if idx == 2 else ""))
                            st.markdown(f"**{medal} {item['name']}**")

                            score = item['score']
                            score_color = "#47e8a3" if score > 0.8 else ("#e8c547" if score > 0.6 else "#e87447")
                            st.markdown(f"<span style='color:{score_color}; font-size:0.78rem; font-weight:600; font-family:DM Mono,monospace;'>{score:.3f}</span>", unsafe_allow_html=True)
                            st.markdown(f"<span class='store-tag'>{item['store']} · {item['level']}</span>", unsafe_allow_html=True)
                            st.markdown("---")
                else:
                    st.warning("No matches found. Try adding more items via bulk_upload.py")