import streamlit as st
import streamlit.components.v1 as components
import requests
from PIL import Image, ImageDraw
import io
import base64
import os
import time

st.set_page_config(layout="wide", page_title="Locus Lens")

GATEWAY_URL = "http://localhost:8000"

# ─── Global Styles (for the main app only) ───────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
    h1, h2, h3 { font-family: 'Syne', sans-serif; }
    .stApp { background: #0a0a0a; color: #f0ede8; }
    .step-badge {
        display: inline-block; background: #e8c547; color: #0d0d0d;
        font-family: 'Syne', sans-serif; font-weight: 800; font-size: 0.75rem;
        padding: 2px 10px; border-radius: 2px; margin-right: 8px; letter-spacing: 0.05em;
    }
    .store-tag { font-size: 0.65rem; color: #666; text-transform: uppercase; letter-spacing: 0.08em; }
    div[data-testid="stButton"] button {
        background: #e8c547; color: #0d0d0d; font-family: 'Syne', sans-serif;
        font-weight: 700; border: none; letter-spacing: 0.05em;
    }
    div[data-testid="stButton"] button:hover { background: #f0d060; }
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


# ─── Loading Screen ───────────────────────────────────────────────────────────
def render_loading_screen(services):
    ready_count  = sum(1 for v in services.values() if v == "ready")
    total_count  = max(len(services), 1)
    progress_pct = int((ready_count / total_count) * 100)

    SERVICE_META = {
        "gateway":       ("🌐", "Gateway",        "API routing &amp; orchestration"),
        "visual_engine": ("🧠", "Vision Engine",   "CLIP &middot; DeepFashion2 &middot; YOLOv8"),
        "qdrant":        ("🗄️",  "Vector Database", "Qdrant similarity search"),
    }

    cards_html = ""
    for i, (key, meta) in enumerate(SERVICE_META.items()):
        icon, name, desc = meta
        status = services.get(key, "loading")
        if status == "ready":
            card_cls, badge_cls, badge_txt = "ready",   "badge-ready",   "&#x25CF; READY"
        elif status == "loading":
            card_cls, badge_cls, badge_txt = "loading", "badge-loading", "&#x25CC; LOADING"
        else:
            card_cls, badge_cls, badge_txt = "error",   "badge-error",   "&#x2715; ERROR"

        cards_html += f"""
        <div class="service-card {card_cls}" style="animation-delay:{i*0.12}s">
            <div class="service-left">
                <div class="icon">{icon}</div>
                <div>
                    <div class="svc-name">{name}</div>
                    <div class="svc-desc">{desc}</div>
                </div>
            </div>
            <div class="badge {badge_cls}">{badge_txt}</div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@300;400&display=swap" rel="stylesheet">
<style>
  *{{ margin:0; padding:0; box-sizing:border-box; }}
  body{{
    background:transparent; color:#f0ede8;
    font-family:'DM Mono',monospace;
    display:flex; justify-content:center;
    padding:24px 12px 16px;
  }}
  .wrap{{ display:flex; flex-direction:column; align-items:center; width:100%; max-width:460px; animation:fadeUp .6s ease both; }}

  @keyframes spin       {{ to{{ transform:rotate(360deg); }} }}
  @keyframes spin-r     {{ to{{ transform:rotate(-360deg); }} }}
  @keyframes glow       {{ 0%,100%{{ box-shadow:0 0 18px rgba(232,197,71,.1); }} 50%{{ box-shadow:0 0 42px rgba(232,197,71,.3); }} }}
  @keyframes fadeUp     {{ from{{ opacity:0; transform:translateY(16px); }} to{{ opacity:1; transform:translateY(0); }} }}
  @keyframes shimmer    {{ 0%{{ background-position:-600px 0; }} 100%{{ background-position:600px 0; }} }}
  @keyframes pulsebadge {{ 0%,100%{{ opacity:1; }} 50%{{ opacity:.4; }} }}
  @keyframes blink      {{ 0%,100%{{ opacity:1; }} 50%{{ opacity:0; }} }}
  @keyframes scan       {{ 0%{{ top:0; opacity:.8; }} 100%{{ top:100%; opacity:0; }} }}

  /* Spinner */
  .spinner{{ position:relative; width:86px; height:86px; margin-bottom:28px; }}
  .r1{{ position:absolute; inset:0; border-radius:50%; border:2.5px solid #1c1c1c; border-top-color:#e8c547; animation:spin 1s linear infinite, glow 2.2s ease-in-out infinite; }}
  .r2{{ position:absolute; inset:14px; border-radius:50%; border:1.5px solid #161616; border-bottom-color:rgba(232,197,71,.3); animation:spin-r 1.7s linear infinite; }}
  .emoji{{ position:absolute; inset:0; display:flex; align-items:center; justify-content:center; font-size:1.35rem; }}

  /* Text */
  .title{{ font-family:'Syne',sans-serif; font-size:1.9rem; font-weight:800; letter-spacing:-.02em; margin-bottom:8px; text-align:center; }}
  .sub{{ font-size:.75rem; color:#484848; text-align:center; line-height:1.8; margin-bottom:36px; }}
  .cursor{{ color:#e8c547; animation:blink 1.1s step-end infinite; }}

  /* Cards */
  .cards{{ width:100%; display:flex; flex-direction:column; gap:8px; margin-bottom:30px; }}
  .service-card{{
    background:#0f0f0f; border:1px solid #1c1c1c; border-radius:10px;
    padding:13px 16px; display:flex; align-items:center; justify-content:space-between;
    position:relative; overflow:hidden; animation:fadeUp .5s ease both;
    transition:border-color .5s, background .5s;
  }}
  .service-card::before{{ content:''; position:absolute; left:0; top:0; bottom:0; width:3px; background:#222; transition:background .5s; }}
  .service-card.ready  {{ border-color:rgba(71,232,163,.2); background:linear-gradient(135deg,#091309 0%,#0f0f0f 55%); }}
  .service-card.ready::before  {{ background:#47e8a3; }}
  .service-card.loading::before{{ background:#e8c547; }}
  .service-card.error::before  {{ background:#e87447; }}
  .service-card.loading::after{{
    content:''; position:absolute; left:0; right:0; height:1px;
    background:linear-gradient(90deg,transparent,rgba(232,197,71,.5),transparent);
    animation:scan 2.2s linear infinite;
  }}
  .service-left{{ display:flex; align-items:center; gap:12px; }}
  .icon{{ font-size:1.15rem; width:24px; text-align:center; flex-shrink:0; }}
  .svc-name{{ font-family:'Syne',sans-serif; font-size:.74rem; font-weight:700; color:#ccc; }}
  .svc-desc{{ font-size:.58rem; color:#363636; margin-top:3px; }}
  .badge{{
    font-size:.56rem; font-weight:700; padding:4px 11px; border-radius:20px;
    letter-spacing:.1em; text-transform:uppercase; white-space:nowrap;
  }}
  .badge-ready  {{ background:rgba(71,232,163,.07); color:#47e8a3; border:1px solid rgba(71,232,163,.25); }}
  .badge-loading{{ background:rgba(232,197,71,.07); color:#e8c547; border:1px solid rgba(232,197,71,.25); animation:pulsebadge 1.5s ease-in-out infinite; }}
  .badge-error  {{ background:rgba(232,116,71,.07); color:#e87447; border:1px solid rgba(232,116,71,.25); }}

  /* Progress */
  .prog-wrap{{ width:100%; margin-bottom:14px; }}
  .prog-meta{{ display:flex; justify-content:space-between; font-size:.56rem; color:#2e2e2e; margin-bottom:7px; letter-spacing:.06em; text-transform:uppercase; }}
  .prog-track{{ height:2px; background:#141414; border-radius:2px; overflow:hidden; }}
  .prog-fill{{ height:100%; border-radius:2px; width:{progress_pct}%; background:linear-gradient(90deg,#e8c547,#f5df7a,#e8c547); background-size:600px 100%; animation:shimmer 1.8s linear infinite; transition:width .8s ease; }}

  .hint{{ font-size:.56rem; color:#222; letter-spacing:.06em; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="spinner">
    <div class="r1"></div>
    <div class="r2"></div>
    <div class="emoji">🔎</div>
  </div>
  <div class="title">Warming Up</div>
  <div class="sub">AI models are loading into memory.<br>This takes 3–5 min on first launch<span class="cursor">_</span></div>
  <div class="cards">{cards_html}</div>
  <div class="prog-wrap">
    <div class="prog-meta"><span>Loading progress</span><span>{ready_count} / {total_count} ready</span></div>
    <div class="prog-track"><div class="prog-fill"></div></div>
  </div>
  <div class="hint">auto-refreshing every 5 seconds</div>
</div>
</body>
</html>"""

    components.html(html, height=500, scrolling=False)


# ─── Gate: show loading screen until ready ───────────────────────────────────
is_ready, services = check_health()

if not is_ready:
    render_loading_screen(services)
    time.sleep(5)
    st.rerun()
    st.stop()


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
                resp  = requests.post(f"{GATEWAY_URL}/detect", files=files, timeout=120)
                if resp.status_code == 200:
                    result = resp.json()
                    st.session_state.detections     = result.get("detections", [])
                    st.session_state.original_image = Image.open(io.BytesIO(new_bytes)).convert("RGB")
                    if not st.session_state.detections:
                        st.warning("No fashion items detected. Try a clearer photo.")
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
        st.markdown(f"**{len(detections)} item{'s' if len(detections)>1 else ''} detected** — select one on the right →")
        st.image(annotated, use_container_width=True)

    with col_select:
        st.markdown("<span class='step-badge'>STEP 2</span> Which item do you want to find?", unsafe_allow_html=True)

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            color = COLORS[i % len(COLORS)]

            patch = orig_img.crop((x1, y1, x2, y2))
            buf   = io.BytesIO()
            patch.save(buf, format="PNG")
            patch_b64 = base64.b64encode(buf.getvalue()).decode()

            is_selected  = st.session_state.selected_idx == i
            border_style = f"3px solid {color}" if is_selected else "1px solid #1a1a1a"
            bg           = "#141414" if is_selected else "#0f0f0f"

            source     = det.get("source", "")
            source_tag = "👗 DeepFashion2" if source == "deepfashion2" else ("👜 Fashionpedia" if source == "yolos_fashionpedia" else "🔍 CLIP")

            st.markdown(f"""
                <div style="border:{border_style}; border-radius:8px; overflow:hidden;
                            margin-bottom:10px; background:{bg};">
                    <img src="data:image/png;base64,{patch_b64}"
                         style="width:100%; display:block; max-height:110px; object-fit:cover;">
                    <div style="padding:8px 12px;">
                        <div style="font-family:Syne,sans-serif; font-weight:700;
                                    font-size:0.78rem; color:{color}; letter-spacing:0.03em;">
                            {i+1}. {det['label'].upper()}
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-top:4px;">
                            <span style="font-size:0.6rem; color:#333;">{source_tag}</span>
                            <span style="font-size:0.6rem; color:#333;">{int(det['score']*100)}% conf</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            btn_label = "✓ Selected" if is_selected else "Select this item"
            if st.button(btn_label, key=f"select_{i}"):
                st.session_state.selected_idx   = i
                st.session_state.search_results = None
                st.rerun()

    # ─── STEP 3: Search ──────────────────────────────────────────────────────
    if st.session_state.selected_idx is not None:
        st.divider()
        selected = detections[st.session_state.selected_idx]
        x1, y1, x2, y2 = selected["bbox"]

        st.markdown(f"<span class='step-badge'>STEP 3</span> Searching for <strong>{selected['label'].upper()}</strong>…", unsafe_allow_html=True)

        col_srch, _ = st.columns([1, 3])
        with col_srch:
            search_btn = st.button("🔍 Find Similar Items", type="primary")

        if search_btn or (st.session_state.search_results is not None):

            if search_btn:
                with st.spinner("⚙️ Processing..."):
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
                result_data       = st.session_state.search_results
                matches           = result_data.get("matches", [])
                debug_image_b64   = result_data.get("debug_image")
                detected_category = result_data.get("detected_category")

                st.divider()

                col_a, col_b, col_c = st.columns([1, 1, 3])
                with col_a:
                    st.markdown("**Your Selection**")
                    st.image(st.session_state.original_image.crop((x1, y1, x2, y2)), use_container_width=True)
                with col_b:
                    st.markdown("**AI Vision**")
                    if debug_image_b64:
                        st.image(Image.open(io.BytesIO(base64.b64decode(debug_image_b64))), use_container_width=True)
                with col_c:
                    df2_label = selected['label'].upper()
                    if detected_category:
                        st.markdown(f"""
                            <div style="background:#0f0f0f; border:1px solid rgba(232,197,71,0.3);
                                        border-left:3px solid #e8c547; border-radius:8px; padding:20px; margin-top:10px;">
                                <div style="font-size:0.6rem; color:#555; letter-spacing:0.12em; text-transform:uppercase; font-family:'DM Mono',monospace;">Detected Item</div>
                                <div style="font-family:Syne,sans-serif; font-size:1.7rem; font-weight:800; color:#e8c547; margin:6px 0 14px;">{df2_label}</div>
                                <div style="font-size:0.6rem; color:#555; letter-spacing:0.12em; text-transform:uppercase; font-family:'DM Mono',monospace;">Search Filter</div>
                                <div style="font-family:Syne,sans-serif; font-size:1.1rem; font-weight:700; color:#888; margin-top:4px;">{detected_category.upper()}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div style="background:#0f0f0f; border:1px solid #1a1a1a;
                                        border-left:3px solid #333; border-radius:8px; padding:20px; margin-top:10px;">
                                <div style="font-size:0.6rem; color:#555; letter-spacing:0.12em; text-transform:uppercase; font-family:'DM Mono',monospace;">Detected Item</div>
                                <div style="font-family:Syne,sans-serif; font-size:1.7rem; font-weight:800; color:#555; margin:6px 0 8px;">{df2_label}</div>
                                <div style="font-size:0.72rem; color:#333; font-family:'DM Mono',monospace;">No category filter — showing all results</div>
                            </div>
                        """, unsafe_allow_html=True)

                st.markdown(f"<h3 style='font-family:Syne,sans-serif; margin-top:28px; font-size:1.2rem;'>🎯 Top Matches <span style='color:#444; font-size:0.8rem; font-family:DM Mono,monospace;'>({len(matches)} found)</span></h3>", unsafe_allow_html=True)

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