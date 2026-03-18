"""
locus_dashboard.py
Locus testing & monitoring dashboard.

Run with:
    pip install streamlit qdrant-client httpx pillow requests
    streamlit run locus_dashboard.py
"""

import io
import base64
import time
import json
import threading
import queue
from datetime import datetime

import requests
import httpx
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from qdrant_client import QdrantClient
from qdrant_client.http import models

# ── Config ────────────────────────────────────────────────────────────────────
import os

GATEWAY_URL     = "http://localhost:8000"
QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "locus_items"

st.set_page_config(
    page_title="Locus Dev Dashboard",
    page_icon="🔍",
    layout="wide",
)

# ── Qdrant client ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_qdrant():
    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(url=QDRANT_URL)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background: #0a0a0a; }
    .stApp { background: #0a0a0a; color: #f0ede8; }
    .metric-card {
        background: #111; border: 1px solid #222;
        border-radius: 10px; padding: 16px 20px; margin: 4px 0;
    }
    .log-box {
        background: #050505; border: 1px solid #1a1a1a;
        border-radius: 8px; padding: 12px; font-family: monospace;
        font-size: 0.75rem; color: #7aab8a; height: 400px;
        overflow-y: auto; white-space: pre-wrap;
    }
    .badge {
        display: inline-block; padding: 2px 8px;
        border-radius: 4px; font-size: 0.7rem; font-weight: 600;
    }
    .cat-shirt    { background: #1a2a3a; color: #6e9ecf; }
    .cat-pants    { background: #1a2535; color: #8aabcf; }
    .cat-dress    { background: #3a1a1a; color: #c97070; }
    .cat-jacket   { background: #1a2a1a; color: #7aab8a; }
    .cat-shoes    { background: #2a2010; color: #c9a96e; }
    .cat-bag      { background: #2a1a10; color: #bf9a7a; }
    .cat-sweater  { background: #2a1a3a; color: #9b8abf; }
    .cat-skirt    { background: #2a2510; color: #c9c06e; }
    .cat-shorts   { background: #102030; color: #8aabcf; }
    .cat-hat      { background: #1a1a1a; color: #aaaaaa; }
    .cat-coat     { background: #1a2a1a; color: #7aab8a; }
    .cat-default  { background: #1a1a1a; color: #888; }
</style>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_monitor, tab_catalogue, tab_search, tab_test = st.tabs([
    "🚀 Scrape & Index", "🗄️ Catalogue", "🔍 Search", "🧪 Test Classify"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SCRAPE & INDEX WITH LIVE MONITORING
# ══════════════════════════════════════════════════════════════════════════════
with tab_monitor:
    st.markdown("### Scrape & Index Monitor")
    st.markdown("Real-time view of what's happening during scraping and indexing.")

    col_form, col_stats = st.columns([1, 1])

    with col_form:
        store_name = st.text_input("Store name", placeholder="Maison 123")
        address    = st.text_input("Address / mall", placeholder="ABC Ashrafieh, Beirut")
        url        = st.text_input("Store URL", placeholder="https://maison123-lb.com/collections/all")

        c1, c2 = st.columns(2)
        scrape_btn = c1.button("🌐 Scrape", use_container_width=True, type="primary")
        index_btn  = c2.button("⚡ Index scraped", use_container_width=True)

    # Session state
    if "scraped_products" not in st.session_state:
        st.session_state.scraped_products = []
    if "index_log"        not in st.session_state:
        st.session_state.index_log = []
    if "index_stats"      not in st.session_state:
        st.session_state.index_stats = {"success": 0, "skipped": 0, "failed": 0, "total": 0}

    # ── Scrape ────────────────────────────────────────────────────────────────
    if scrape_btn:
        if not url.strip():
            st.error("Enter a URL first.")
        else:
            with st.spinner("Scraping…"):
                try:
                    resp = requests.post(
                        f"{GATEWAY_URL}/scrape",
                        json={"url": url, "max_products": 0},
                        timeout=30,
                    )
                    data = resp.json()
                    st.session_state.scraped_products = data.get("products", [])
                    st.session_state.index_log        = []
                    st.session_state.index_stats      = {"success": 0, "skipped": 0, "failed": 0, "total": 0}
                    st.success(f"✅ Found {len(st.session_state.scraped_products)} products")
                except Exception as e:
                    st.error(f"Scrape failed: {e}")

    with col_stats:
        stats = st.session_state.index_stats
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total",   stats["total"])
        s2.metric("✅ OK",   stats["success"])
        s3.metric("⏭ Skip", stats["skipped"])
        s4.metric("❌ Fail", stats["failed"])

    # ── Product preview grid ───────────────────────────────────────────────────
    products = st.session_state.scraped_products
    if products:
        st.markdown(f"**{len(products)} products scraped** — preview (first 20):")
        cols = st.columns(5)
        for i, p in enumerate(products[:20]):
            with cols[i % 5]:
                img_url = p.get("image_url") or (p.get("image_urls") or [""])[0]
                if img_url:
                    st.image(img_url, use_container_width=True)
                st.caption(p.get("name", "")[:40])

    # ── Box preview (dry run on first N products) ──────────────────────────────
    if products and st.button("🔬 Preview boxes (first 8 products)", key="preview_btn"):
        st.markdown("#### Box preview — what the indexer will crop")
        st.caption("Gold box = selected crop · Grey boxes = other detections · No box = full image used")

        VISUAL_URL = GATEWAY_URL.replace("8000", "8001")
        preview_products = products[:8]
        preview_cols = st.columns(4)

        for i, p in enumerate(preview_products):
            img_url = p.get("image_url") or (p.get("image_urls") or [""])[0]
            name    = p.get("name", "")

            with preview_cols[i % 4]:
                if not img_url:
                    st.caption(f"No image: {name[:30]}")
                    continue
                try:
                    img_resp  = requests.get(img_url, timeout=10)
                    img_bytes = img_resp.content

                    dbg_resp = requests.post(
                        f"{VISUAL_URL}/debug-index",
                        files={"file": ("product.jpg", img_bytes, "image/jpeg")},
                        data={"title": name},
                        timeout=60,
                    )
                    dbg = dbg_resp.json()

                    if dbg.get("debug_image"):
                        import base64 as _b64
                        img_data = _b64.b64decode(dbg["debug_image"])
                        st.image(img_data, use_container_width=True)

                    cat = dbg.get("category") or "skipped"
                    skip_reason = dbg.get("skip_reason", "")
                    box_src     = dbg.get("box_source", "")
                    color       = "#c9a96e" if not dbg.get("skipped") else "#c97070"

                    st.markdown(
                        f'<span style="color:{color};font-size:0.72rem;font-weight:600">'
                        f'{"⏭ " + skip_reason if dbg.get("skipped") else "✓ " + cat}'
                        f'</span> <span style="color:#555;font-size:0.65rem">{box_src}</span>',
                        unsafe_allow_html=True,
                    )
                    st.caption(name[:40])

                except Exception as e:
                    st.caption(f"Error: {str(e)[:40]}")


    if index_btn:
        if not products:
            st.error("Scrape first.")
        elif not store_name.strip():
            st.error("Enter store name.")
        else:
            log_placeholder  = st.empty()
            prog_placeholder = st.empty()

            log_lines  = []
            success    = 0
            skipped    = 0
            failed     = 0
            total      = len(products)
            CHUNK_SIZE = 10

            for start in range(0, total, CHUNK_SIZE):
                chunk     = products[start:start + CHUNK_SIZE]
                batch_num = start // CHUNK_SIZE + 1
                total_bat = (total + CHUNK_SIZE - 1) // CHUNK_SIZE

                prog_placeholder.progress(
                    start / total,
                    text=f"Batch {batch_num}/{total_bat} · {start}/{total} products"
                )

                try:
                    resp = requests.post(
                        f"{GATEWAY_URL}/add-bulk-batch",
                        json={
                            "items": [
                                {
                                    "name":       p.get("name", "Product"),
                                    "store":      store_name,
                                    "mall":       address,
                                    "image_urls": p.get("image_urls") or ([p["image_url"]] if p.get("image_url") else []),
                                    "image_url":  p.get("image_url", ""),
                                    "price":      p.get("price", ""),
                                    "category":   p.get("category", ""),
                                }
                                for p in chunk
                            ]
                        },
                        timeout=300,
                    )
                    data     = resp.json()
                    success += data.get("success", 0)
                    skipped += data.get("skipped", 0)
                    for f in data.get("failed", []):
                        failed += 1
                        log_lines.append(f"❌ FAILED  {f['item']}: {f.get('error','')[:60]}")

                    # Log each item result
                    for item in chunk:
                        name = item.get("name", "")[:50]
                        log_lines.append(f"✅ OK      {name}")

                except Exception as e:
                    failed += len(chunk)
                    log_lines.append(f"❌ BATCH {batch_num} ERROR: {e}")

                # Update live log (last 40 lines)
                st.session_state.index_stats = {
                    "success": success, "skipped": skipped,
                    "failed": failed,   "total":   total,
                }
                log_text = "\n".join(log_lines[-40:])
                log_placeholder.markdown(
                    f'<div class="log-box">{log_text}</div>',
                    unsafe_allow_html=True,
                )

            prog_placeholder.progress(1.0, text="Done!")
            st.success(f"Indexed {success} · Skipped {skipped} · Failed {failed}")
            st.session_state.index_log = log_lines


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CATALOGUE BROWSER
# ══════════════════════════════════════════════════════════════════════════════
with tab_catalogue:
    st.markdown("### Catalogue Browser")
    st.markdown("Browse everything indexed in Qdrant with bounding boxes and metadata.")

    client = get_qdrant()

    # Stats
    try:
        info   = client.get_collection(COLLECTION_NAME)
        n_pts  = info.points_count
        st.markdown(f"**{n_pts} points** in `{COLLECTION_NAME}`")
    except Exception as e:
        st.error(f"Cannot reach Qdrant: {e}")
        st.stop()

    # Filters
    cf1, cf2, cf3 = st.columns([2, 2, 1])
    filter_store = cf1.text_input("Filter by store", placeholder="Maison 123")
    filter_cat   = cf2.selectbox("Filter by category", [
        "all", "shirt", "sweater", "jacket", "coat", "dress",
        "jumpsuit", "skirt", "pants", "shorts", "shoes", "bag",
        "glasses", "hat", "watch", "scarf",
    ])
    page_size = cf3.selectbox("Per page", [12, 24, 48], index=1)

    if "cat_offset" not in st.session_state:
        st.session_state.cat_offset = 0

    # Build filter
    must_conditions = []
    if filter_store.strip():
        must_conditions.append(models.FieldCondition(
            key="store_name", match=models.MatchValue(value=filter_store.strip())
        ))
    if filter_cat != "all":
        must_conditions.append(models.FieldCondition(
            key="category_tag", match=models.MatchValue(value=filter_cat)
        ))
    scroll_filter = models.Filter(must=must_conditions) if must_conditions else None

    # Fetch
    try:
        results, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=page_size,
            offset=st.session_state.cat_offset,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        st.error(f"Scroll failed: {e}")
        results = []

    # Deduplicate by product_id (2 points per product — normal + dark)
    seen = {}
    for pt in results:
        p  = pt.payload
        pid = p.get("product_id", str(pt.id))
        if pid not in seen:
            seen[pid] = p
    unique = list(seen.values())

    # Grid
    CAT_COLORS = {
        "shirt": "#6e9ecf", "sweater": "#9b8abf", "jacket": "#7aab8a",
        "coat": "#7aab8a", "dress": "#c97070", "jumpsuit": "#c9a96e",
        "skirt": "#c9c06e", "pants": "#8aabcf", "shorts": "#8aabcf",
        "shoes": "#c9a96e", "bag": "#bf9a7a", "glasses": "#aaa",
        "hat": "#aaa", "watch": "#c9c96e", "scarf": "#bf9a7a",
    }

    cols = st.columns(4)
    for i, p in enumerate(unique):
        with cols[i % 4]:
            img_url = p.get("image_url", "")
            cat     = p.get("category_tag", "unknown")
            color   = CAT_COLORS.get(cat, "#888")

            if img_url:
                st.image(img_url, use_container_width=True)
            else:
                st.markdown("📦 *(no image)*")

            st.markdown(
                f'<span class="badge" style="background:{color}22;color:{color};'
                f'border:1px solid {color}44">{cat}</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"**{p.get('name','')[:45]}**")
            st.caption(f"{p.get('store_name','')} · {p.get('price','')}")

    # Pagination
    p1, p2, p3 = st.columns([1, 2, 1])
    if p1.button("← Prev", disabled=st.session_state.cat_offset == 0):
        st.session_state.cat_offset = max(0, st.session_state.cat_offset - page_size)
        st.rerun()
    p2.markdown(
        f"<div style='text-align:center;color:#666;font-size:0.8rem'>"
        f"offset {st.session_state.cat_offset}</div>",
        unsafe_allow_html=True,
    )
    if p3.button("Next →", disabled=len(unique) < page_size):
        st.session_state.cat_offset += page_size
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — VISUAL SEARCH TEST
# ══════════════════════════════════════════════════════════════════════════════
with tab_search:
    st.markdown("### Visual Search Test")
    st.markdown("Upload a photo, select a bounding box, search Qdrant.")

    uploaded = st.file_uploader("Upload query image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded:
        img_bytes = uploaded.read()
        image     = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        W, H      = image.size

        # Detect
        with st.spinner("Detecting items…"):
            try:
                resp       = requests.post(
                    f"{GATEWAY_URL}/detect",
                    files={"file": (uploaded.name, img_bytes, "image/jpeg")},
                    timeout=60,
                )
                detections = resp.json().get("detections", [])
            except Exception as e:
                st.error(f"Detect failed: {e}")
                detections = []

        # Draw boxes on image
        draw  = ImageDraw.Draw(image.copy())
        img_annotated = image.copy()
        draw  = ImageDraw.Draw(img_annotated)

        BOX_COLORS = [
            "#c9a96e", "#7aab8a", "#c97070", "#9b8abf",
            "#6e9ecf", "#c9c06e", "#bf9a7a", "#aaaaaa",
        ]
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            color = BOX_COLORS[idx % len(BOX_COLORS)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label_text = f"{idx}: {det.get('search_label','?')} {det['score']:.2f}"
            draw.rectangle([x1, y1 - 18, x1 + len(label_text) * 7, y1], fill=color)
            draw.text((x1 + 2, y1 - 16), label_text, fill="#000")

        col_img, col_boxes = st.columns([2, 1])
        with col_img:
            st.image(img_annotated, caption="Detected boxes", use_container_width=True)
        with col_boxes:
            if detections:
                options = {
                    f"{i}: {d.get('search_label','?')} (conf {d['score']:.2f})": i
                    for i, d in enumerate(detections)
                }
                selected_label = st.radio("Select box to search with:", list(options.keys()))
                selected_idx   = options[selected_label]
                selected_box   = detections[selected_idx]

                st.markdown(f"**Label:** `{selected_box.get('label','?')}`")
                st.markdown(f"**Search label:** `{selected_box.get('search_label','?')}`")
                st.markdown(f"**Confidence:** `{selected_box['score']:.3f}`")
                st.markdown(f"**Source:** `{selected_box.get('source','?')}`")

                x1, y1, x2, y2 = selected_box["bbox"]
                crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
                st.image(crop, caption="Crop that will be searched", width=150)

                search_btn = st.button("🔍 Search this crop", type="primary")
            else:
                st.info("No boxes detected.")
                search_btn = False

        if detections and search_btn:
            x1, y1, x2, y2 = selected_box["bbox"]
            with st.spinner("Searching…"):
                try:
                    resp = requests.post(
                        f"{GATEWAY_URL}/search",
                        files={"file": (uploaded.name, img_bytes, "image/jpeg")},
                        data={
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "search_label": selected_box.get("search_label", ""),
                        },
                        timeout=60,
                    )
                    results = resp.json()
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    results = {}

            matches = results.get("matches", [])
            det_cat = results.get("detected_category", "?")
            st.markdown(f"**Detected category:** `{det_cat}` · **{len(matches)} matches**")

            if matches:
                mcols = st.columns(5)
                for i, m in enumerate(matches[:15]):
                    with mcols[i % 5]:
                        if m.get("image_url"):
                            st.image(m["image_url"], use_container_width=True)
                        st.caption(f"**{m['name'][:35]}**")
                        st.caption(f"{m['store_name']} · {m.get('price','')}")
                        st.caption(f"Score: `{m['score']:.3f}`")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TEST CLASSIFY
# ══════════════════════════════════════════════════════════════════════════════
with tab_test:
    st.markdown("### Test Title Classifier")
    st.markdown("Type any product title and see how Signal 1 (title) classifies it.")

    test_title = st.text_input("Product title", placeholder="Adidas Wide Leg Track Pant Black")

    if st.button("Classify") and test_title.strip():
        with st.spinner("Classifying…"):
            try:
                resp = requests.post(
                    f"{GATEWAY_URL.replace('8000','8001')}/classify-text",
                    json={"title": test_title},
                    timeout=10,
                )
                data = resp.json()
                st.markdown(f"**Category:** `{data.get('category','None')}`")
                st.markdown(f"**Confidence:** `{data.get('confidence', 0):.4f}`")
            except Exception as e:
                st.error(f"Failed: {e}")

    st.divider()
    st.markdown("### Test Image Classification")
    st.markdown("Upload an image and see all 3 signals for a given title.")

    tc1, tc2 = st.columns([1, 2])
    with tc1:
        test_img  = st.file_uploader("Image", type=["jpg","jpeg","png","webp"], key="test_img")
        test_name = st.text_input("Product title", placeholder="Adidas Hoodie Black", key="test_name")
        test_go   = st.button("Run 3-signal analysis", type="primary")

    with tc2:
        if test_go and test_img and test_name:
            img_bytes = test_img.read()
            with st.spinner("Running…"):
                try:
                    resp = requests.post(
                        f"{GATEWAY_URL.replace('8000','8001')}/index-image",
                        files={"file": (test_img.name, img_bytes, "image/jpeg")},
                        data={"title": test_name},
                        timeout=90,
                    )
                    data = resp.json()

                    if data.get("skipped"):
                        st.warning(f"⏭ Skipped: `{data.get('skip_reason')}`")
                    else:
                        cat        = data.get("category", "?")
                        box_source = data.get("box_source", "?")
                        color      = CAT_COLORS.get(cat, "#888") if 'CAT_COLORS' in dir() else "#888"
                        st.markdown(
                            f'<span class="badge" style="background:{color}22;color:{color};'
                            f'border:1px solid {color}44;font-size:1rem;padding:6px 14px">'
                            f'{cat}</span>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"**Box source:** `{box_source}`")
                        st.markdown(f"**Vector dims:** `{len(data.get('vector_normal',[]))}`")
                        st.success("Indexed successfully (not saved to Qdrant — test only)")

                except Exception as e:
                    st.error(f"Failed: {e}")