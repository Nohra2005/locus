import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os

st.set_page_config(layout="wide", page_title="Locus Lens")

# ─── Styling ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Inter:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Syne', sans-serif; }

    .stApp { background: #0d0d0d; color: #f0ede8; }

    .detection-box {
        border: 2px solid #f0ede8;
        border-radius: 4px;
        padding: 0;
        margin-bottom: 8px;
        cursor: pointer;
        overflow: hidden;
        transition: border-color 0.2s, transform 0.15s;
        background: #1a1a1a;
    }
    .detection-box:hover { border-color: #e8c547; transform: scale(1.02); }
    .detection-box.selected { border-color: #e8c547; border-width: 3px; }
    .det-label {
        font-family: 'Syne', sans-serif;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 4px 8px;
        background: #e8c547;
        color: #0d0d0d;
    }
    .score-pill {
        font-size: 0.65rem;
        color: #888;
        padding: 2px 8px 4px;
    }
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
    .result-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 6px;
        padding: 10px;
        margin-bottom: 8px;
        transition: border-color 0.2s;
    }
    .result-card:hover { border-color: #444; }
    .result-rank-1 { border-color: #e8c547 !important; }
    .store-tag {
        font-size: 0.65rem;
        color: #888;
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
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("<h1 style='font-size:2.5rem; margin-bottom:0'>🔎 LOCUS LENS</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#888; margin-top:4px; font-size:0.95rem;'>Upload a photo — AI detects every item — click what you want to find</p>", unsafe_allow_html=True)
st.divider()

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
    
    # If a new file was uploaded, reset state
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
                resp = requests.post("http://localhost:8000/detect", files=files, timeout=120)
                
                if resp.status_code == 200:
                    result = resp.json()
                    st.session_state.detections = result.get("detections", [])
                    st.session_state.original_image = Image.open(io.BytesIO(new_bytes)).convert("RGB")
                    
                    if not st.session_state.detections:
                        st.warning("No fashion items detected. Try a clearer photo with better lighting.")
                else:
                    st.error(f"Detection failed: {resp.status_code}")
            except Exception as e:
                st.error(f"Connection error: {e}")

# ─── STEP 2 RESULTS: Show annotated image + click-to-select grid ─────────────
if st.session_state.detections and st.session_state.original_image:
    detections = st.session_state.detections
    orig_img = st.session_state.original_image

    # Draw bounding boxes on the main image
    annotated = orig_img.copy()
    draw = ImageDraw.Draw(annotated)
    
    COLORS = ["#e8c547", "#47c5e8", "#e847a3", "#47e8a3", "#e87447", "#a347e8"]
    
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        color = COLORS[i % len(COLORS)]
        # Draw box with number label
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # Label badge background
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
            
            # Crop thumbnail for each detected item
            patch = orig_img.crop((x1, y1, x2, y2))
            patch_buf = io.BytesIO()
            patch.save(patch_buf, format="PNG")
            patch_b64 = base64.b64encode(patch_buf.getvalue()).decode()
            
            is_selected = st.session_state.selected_idx == i
            border_style = f"3px solid {color}" if is_selected else f"2px solid #2a2a2a"
            bg = "#1f1f1f" if is_selected else "#141414"
            
            # Show thumbnail + label in a styled card
            st.markdown(f"""
                <div style="border:{border_style}; border-radius:6px; overflow:hidden; margin-bottom:10px; background:{bg}; cursor:pointer;">
                    <img src="data:image/png;base64,{patch_b64}" style="width:100%; display:block; max-height:120px; object-fit:cover;">
                    <div style="padding:6px 10px;">
                        <span style="font-family:Syne,sans-serif; font-weight:700; font-size:0.8rem; color:{color};">
                            {i+1}. {det['label'].upper()}
                        </span>
                        <span style="font-size:0.65rem; color:#666; margin-left:8px;">
                            {int(det['score']*100)}% confidence
                        </span>
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
        
        st.markdown(f"<span class='step-badge'>STEP 3</span> Searching for **{selected['label'].upper()}** in the inventory…", unsafe_allow_html=True)
        
        col_srch, _ = st.columns([1, 3])
        with col_srch:
            search_btn = st.button("🔍 Find Similar Items", type="primary")
        
        if search_btn or (st.session_state.search_results is not None):
            
            if search_btn:  # Only re-fetch if button was clicked
                with st.spinner("⚙️ AI processing your selection..."):
                    try:
                        files = {"file": ("image.png", st.session_state.uploaded_bytes, "image/png")}
                        data = {
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2
                        }
                        resp = requests.post(
                            "http://localhost:8000/search",
                            files=files,
                            data=data,
                            timeout=60
                        )
                        
                        if resp.status_code == 200:
                            st.session_state.search_results = resp.json()
                        else:
                            st.error(f"Search failed: {resp.status_code} — {resp.text}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")

            # ─── RESULTS ─────────────────────────────────────────────────────
            if st.session_state.search_results:
                result_data = st.session_state.search_results
                matches = result_data.get("matches", [])
                debug_image_b64 = result_data.get("debug_image")
                detected_category = result_data.get("detected_category")

                st.divider()

                # AI Vision + Category banner
                col_a, col_b, col_c = st.columns([1, 1, 3])
                
                with col_a:
                    st.markdown("**Your Selection**")
                    patch = st.session_state.original_image.crop((x1, y1, x2, y2))
                    st.image(patch, use_container_width=True)

                with col_b:
                    st.markdown("**AI Vision**")
                    if debug_image_b64:
                        ai_img_data = base64.b64decode(debug_image_b64)
                        ai_img = Image.open(io.BytesIO(ai_img_data))
                        st.image(ai_img, use_container_width=True)

                with col_c:
                    if detected_category:
                        st.markdown(f"""
                            <div style="background:#1a1a1a; border:1px solid #e8c547; border-radius:6px; padding:20px; margin-top:10px;">
                                <div style="font-family:Syne,sans-serif; font-size:0.7rem; color:#888; letter-spacing:0.1em; text-transform:uppercase;">AI Identified</div>
                                <div style="font-family:Syne,sans-serif; font-size:2rem; font-weight:800; color:#e8c547; margin:4px 0;">{detected_category.upper()}</div>
                                <div style="font-size:0.8rem; color:#666;">Category filter is active — results are narrowed to this type</div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div style="background:#1a1a1a; border:1px solid #444; border-radius:6px; padding:20px; margin-top:10px;">
                                <div style="font-family:Syne,sans-serif; font-size:0.7rem; color:#888; letter-spacing:0.1em; text-transform:uppercase;">AI Status</div>
                                <div style="font-family:Syne,sans-serif; font-size:1.5rem; font-weight:800; color:#888; margin:4px 0;">UNCLASSIFIED</div>
                                <div style="font-size:0.8rem; color:#555;">Showing results across all categories</div>
                            </div>
                        """, unsafe_allow_html=True)

                # Results grid
                st.markdown(f"<h3 style='font-family:Syne,sans-serif; margin-top:24px;'>🎯 Top Matches ({len(matches)} found)</h3>", unsafe_allow_html=True)
                
                if matches:
                    cols = st.columns(5)
                    for idx, item in enumerate(matches):
                        with cols[idx % 5]:
                            local_path = os.path.join("demo_images", item['image_filename'])
                            
                            card_class = "result-rank-1" if idx == 0 else ""
                            
                            if os.path.exists(local_path):
                                st.image(local_path, use_container_width=True)
                            
                            medal = "🥇" if idx == 0 else ("🥈" if idx == 1 else ("🥉" if idx == 2 else ""))
                            st.markdown(f"**{medal} {item['name']}**")
                            
                            score = item['score']
                            score_color = "#47e8a3" if score > 0.8 else ("#e8c547" if score > 0.6 else "#e87447")
                            st.markdown(f"<span style='color:{score_color}; font-size:0.8rem; font-weight:600;'>{score:.3f}</span>", unsafe_allow_html=True)
                            st.markdown(f"<span class='store-tag'>{item['store']} • {item['level']}</span>", unsafe_allow_html=True)
                            st.markdown("---")
                else:
                    st.warning("No matches found in inventory. Try adding more items via bulk_upload.py")