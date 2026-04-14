"""
locus_dashboard.py — Locus developer dashboard.

Run with:
    pip install streamlit qdrant-client pillow requests python-dotenv
    streamlit run locus_dashboard.py
"""

import io
import base64
import time
import os

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import requests
import streamlit as st
from PIL import Image, ImageDraw
from qdrant_client import QdrantClient
from qdrant_client.http import models

# ── Config ────────────────────────────────────────────────────────────────────
GATEWAY_URL        = "http://localhost:8000"
VISUAL_URL         = "http://localhost:8001"
QDRANT_URL         = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY     = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME    = "locus_items"
SKIPPED_COLLECTION = "locus_skipped"

st.set_page_config(page_title="Locus Dev Dashboard", page_icon="🔍", layout="wide")

@st.cache_resource
def get_qdrant():
    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(url=QDRANT_URL)

# ── Helpers ───────────────────────────────────────────────────────────────────
def fetch_pending_suggestions():
    try:
        r = requests.get(f"{GATEWAY_URL}/whitelist-suggestions?status=pending", timeout=5)
        return r.json().get("suggestions", []) if r.ok else []
    except Exception:
        return []

def fetch_skipped_count():
    try:
        r = requests.get(f"{GATEWAY_URL}/skipped-products?limit=1", timeout=5)
        return r.json().get("total", 0) if r.ok else 0
    except Exception:
        return 0

CAT_COLORS = {
    "top": "#6e9ecf", "sports_bra": "#c97070", "pants": "#8aabcf",
    "leggings": "#9b8abf", "shorts": "#8aabcf", "skirt": "#c9c06e",
    "dress": "#c97070", "sweater": "#9b8abf", "jacket": "#7aab8a",
    "shoes": "#c9a96e", "hat": "#aaaaaa", "bag": "#bf9a7a", "jumpsuit": "#c9a96e",
}
REASON_COLORS = {
    "no_consensus": "#c9a96e",
    "no_box_found": "#c97070",
    "not_fashion":  "#555555",
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: #0a0a0a; color: #f0ede8; }
    .log-box {
        background: #050505; border: 1px solid #1a1a1a; border-radius: 8px;
        padding: 12px; font-family: monospace; font-size: 0.75rem;
        color: #7aab8a; height: 400px; overflow-y: auto; white-space: pre-wrap;
    }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; }
    .alert-banner {
        background: rgba(201,112,112,0.08); border: 1px solid rgba(201,112,112,0.3);
        border-left: 4px solid #c97070; border-radius: 8px; padding: 14px 18px; margin-bottom: 20px;
    }
    .pending-card {
        background: #111; border: 1px solid #222; border-left: 4px solid #c9a96e;
        border-radius: 8px; padding: 16px 20px; margin-bottom: 12px;
    }
    .approved-card {
        background: rgba(122,171,138,0.05); border: 1px solid rgba(122,171,138,0.2);
        border-left: 4px solid #7aab8a; border-radius: 8px; padding: 12px 18px; margin-bottom: 8px;
    }
    .rejected-card {
        background: rgba(100,100,100,0.05); border: 1px solid #222; border-left: 4px solid #444;
        border-radius: 8px; padding: 12px 18px; margin-bottom: 8px; opacity: 0.6;
    }
</style>
""", unsafe_allow_html=True)

# ── Notification banner ───────────────────────────────────────────────────────
pending_suggestions = fetch_pending_suggestions()
skipped_count       = fetch_skipped_count()

if len(pending_suggestions) > 0 or skipped_count > 0:
    parts = []
    if pending_suggestions:
        parts.append(f"**{len(pending_suggestions)} whitelist suggestion{'s' if len(pending_suggestions) > 1 else ''} pending review**")
    if skipped_count > 0:
        parts.append(f"**{skipped_count} skipped products** awaiting fixes")
    st.markdown(
        f'<div class="alert-banner">⚠️ &nbsp; {" · ".join(parts)} — see the <strong>📋 Whitelist Review</strong> tab</div>',
        unsafe_allow_html=True,
    )

review_label = f"📋 Whitelist Review {'🔴' if pending_suggestions else ''}"

tab_monitor, tab_catalogue, tab_search, tab_test, tab_review = st.tabs([
    "🚀 Scrape & Index", "🗄️ Catalogue", "🔍 Search", "🧪 Test Classify", review_label,
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SCRAPE & INDEX
# ══════════════════════════════════════════════════════════════════════════════
with tab_monitor:
    st.markdown("### Scrape & Index Monitor")
    col_form, col_stats = st.columns([1, 1])

    with col_form:
        store_name = st.text_input("Store name", placeholder="Maison 123")
        address    = st.text_input("Address / mall", placeholder="ABC Ashrafieh, Beirut")
        url        = st.text_input("Store URL", placeholder="https://yourstore.com/collections/all")
        c1, c2     = st.columns(2)
        scrape_btn = c1.button("🌐 Scrape", use_container_width=True, type="primary")
        index_btn  = c2.button("⚡ Index scraped", use_container_width=True)

    for key, default in [("scraped_products", []), ("index_log", []), ("index_stats", {"success": 0, "skipped": 0, "failed": 0, "total": 0})]:
        if key not in st.session_state:
            st.session_state[key] = default

    if scrape_btn:
        if not url.strip():
            st.error("Enter a URL first.")
        else:
            with st.spinner("Scraping…"):
                try:
                    resp = requests.post(f"{GATEWAY_URL}/scrape", json={"url": url, "max_products": 0}, timeout=30)
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

    if products and st.button("🔬 Preview boxes (first 8)", key="preview_btn"):
        st.markdown("#### Box preview")
        preview_cols = st.columns(4)
        for i, p in enumerate(products[:8]):
            img_url = p.get("image_url") or (p.get("image_urls") or [""])[0]
            name    = p.get("name", "")
            with preview_cols[i % 4]:
                if not img_url:
                    st.caption(f"No image: {name[:30]}")
                    continue
                try:
                    img_bytes = requests.get(img_url, timeout=10).content
                    dbg = requests.post(f"{VISUAL_URL}/debug-index", files={"file": ("product.jpg", img_bytes, "image/jpeg")}, data={"title": name}, timeout=60).json()
                    if dbg.get("debug_image"):
                        st.image(base64.b64decode(dbg["debug_image"]), use_container_width=True)
                    cat   = dbg.get("category") or "skipped"
                    color = "#c9a96e" if not dbg.get("skipped") else "#c97070"
                    label = ("⏭ " + dbg.get("skip_reason","")) if dbg.get("skipped") else ("✓ " + cat)
                    st.markdown(f'<span style="color:{color};font-size:0.72rem;font-weight:600">{label}</span> <span style="color:#555;font-size:0.65rem">{dbg.get("box_source","")}</span>', unsafe_allow_html=True)
                    st.caption(name[:40])
                except Exception as e:
                    st.caption(f"Error: {str(e)[:40]}")

    if index_btn:
        if not products:
            st.error("Scrape first.")
        elif not store_name.strip():
            st.error("Enter store name.")
        else:
            log_ph  = st.empty()
            prog_ph = st.empty()
            log_lines = []
            success = skipped_n = failed = 0
            total = len(products)
            CHUNK = 10
            for start in range(0, total, CHUNK):
                chunk     = products[start:start + CHUNK]
                batch_num = start // CHUNK + 1
                total_bat = (total + CHUNK - 1) // CHUNK
                prog_ph.progress(start / total, text=f"Batch {batch_num}/{total_bat} · {start}/{total}")
                try:
                    resp = requests.post(f"{GATEWAY_URL}/add-bulk-batch", json={"items": [{"name": p.get("name","Product"), "store": store_name, "mall": address, "image_urls": p.get("image_urls") or ([p["image_url"]] if p.get("image_url") else []), "image_url": p.get("image_url",""), "price": p.get("price",""), "category": p.get("category","")} for p in chunk]}, timeout=300)
                    data = resp.json()
                    success  += data.get("success", 0)
                    skipped_n += data.get("skipped", 0)
                    for f in data.get("failed", []):
                        failed += 1
                        log_lines.append(f"❌ FAILED  {f['item']}: {f.get('error','')[:60]}")
                    for item in chunk:
                        log_lines.append(f"✅ OK      {item.get('name','')[:50]}")
                except Exception as e:
                    failed += len(chunk)
                    log_lines.append(f"❌ BATCH {batch_num} ERROR: {e}")
                st.session_state.index_stats = {"success": success, "skipped": skipped_n, "failed": failed, "total": total}
                log_ph.markdown(f'<div class="log-box">{chr(10).join(log_lines[-40:])}</div>', unsafe_allow_html=True)
            prog_ph.progress(1.0, text="Done!")
            st.success(f"Indexed {success} · Skipped {skipped_n} · Failed {failed}")
            if skipped_n > 0:
                st.info(f"ℹ️ {skipped_n} products skipped — check the **📋 Whitelist Review** tab.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CATALOGUE
# ══════════════════════════════════════════════════════════════════════════════
with tab_catalogue:
    st.markdown("### Catalogue Browser")
    client = get_qdrant()
    try:
        info  = client.get_collection(COLLECTION_NAME)
        st.markdown(f"**{info.points_count} points** in `{COLLECTION_NAME}`")
    except Exception as e:
        st.error(f"Cannot reach Qdrant: {e}")
        st.stop()

    cf1, cf2, cf3 = st.columns([2, 2, 1])
    filter_store = cf1.text_input("Filter by store", placeholder="Maison 123")
    filter_cat   = cf2.selectbox("Filter by category", ["all"] + list(CAT_COLORS.keys()))
    page_size    = cf3.selectbox("Per page", [12, 24, 48], index=1)

    if "cat_offset" not in st.session_state:
        st.session_state.cat_offset = 0

    must = []
    if filter_store.strip():
        must.append(models.FieldCondition(key="store_name", match=models.MatchValue(value=filter_store.strip())))
    if filter_cat != "all":
        must.append(models.FieldCondition(key="category_tag", match=models.MatchValue(value=filter_cat)))
    scroll_filter = models.Filter(must=must) if must else None

    try:
        results, _ = client.scroll(collection_name=COLLECTION_NAME, scroll_filter=scroll_filter, limit=page_size, offset=st.session_state.cat_offset, with_payload=True, with_vectors=False)
    except Exception as e:
        st.error(f"Scroll failed: {e}")
        results = []

    seen = {}
    for pt in results:
        p = pt.payload
        pid = p.get("product_id", str(pt.id))
        if pid not in seen:
            seen[pid] = p
    unique = list(seen.values())

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
            st.markdown(f'<span class="badge" style="background:{color}22;color:{color};border:1px solid {color}44">{cat}</span>', unsafe_allow_html=True)
            st.caption(f"**{p.get('name','')[:45]}**")
            st.caption(f"{p.get('store_name','')} · {p.get('price','')}")

    p1, p2, p3 = st.columns([1, 2, 1])
    if p1.button("← Prev", disabled=st.session_state.cat_offset == 0):
        st.session_state.cat_offset = max(0, st.session_state.cat_offset - page_size)
        st.rerun()
    p2.markdown(f"<div style='text-align:center;color:#666;font-size:0.8rem'>offset {st.session_state.cat_offset}</div>", unsafe_allow_html=True)
    if p3.button("Next →", disabled=len(unique) < page_size):
        st.session_state.cat_offset += page_size
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — VISUAL SEARCH
# ══════════════════════════════════════════════════════════════════════════════
with tab_search:
    st.markdown("### Visual Search Test")
    uploaded = st.file_uploader("Upload query image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded:
        img_bytes = uploaded.read()
        image     = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        with st.spinner("Detecting items…"):
            try:
                resp       = requests.post(f"{GATEWAY_URL}/detect", files={"file": (uploaded.name, img_bytes, "image/jpeg")}, timeout=60)
                detections = resp.json().get("detections", [])
            except Exception as e:
                st.error(f"Detect failed: {e}")
                detections = []

        img_annotated = image.copy()
        draw          = ImageDraw.Draw(img_annotated)
        BOX_COLORS    = ["#c9a96e", "#7aab8a", "#c97070", "#9b8abf", "#6e9ecf", "#c9c06e", "#bf9a7a", "#aaaaaa"]
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            color = BOX_COLORS[idx % len(BOX_COLORS)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            lbl = f"{idx}: {det.get('search_label','?')} {det['score']:.2f}"
            draw.rectangle([x1, y1 - 18, x1 + len(lbl) * 7, y1], fill=color)
            draw.text((x1 + 2, y1 - 16), lbl, fill="#000")

        col_img, col_boxes = st.columns([2, 1])
        with col_img:
            st.image(img_annotated, caption="Detected boxes", use_container_width=True)
        with col_boxes:
            if detections:
                options        = {f"{i}: {d.get('search_label','?')} (conf {d['score']:.2f})": i for i, d in enumerate(detections)}
                selected_label = st.radio("Select box:", list(options.keys()))
                selected_box   = detections[options[selected_label]]
                st.markdown(f"**Label:** `{selected_box.get('label','?')}`")
                st.markdown(f"**Search label:** `{selected_box.get('search_label','?')}`")
                st.markdown(f"**Confidence:** `{selected_box['score']:.3f}`")
                st.markdown(f"**Source:** `{selected_box.get('source','?')}`")
                x1, y1, x2, y2 = selected_box["bbox"]
                st.image(image.crop((int(x1), int(y1), int(x2), int(y2))), caption="Crop", width=150)
                search_btn = st.button("🔍 Search this crop", type="primary")
            else:
                st.info("No boxes detected.")
                search_btn = False

        if detections and search_btn:
            x1, y1, x2, y2 = selected_box["bbox"]
            with st.spinner("Searching…"):
                try:
                    resp    = requests.post(f"{GATEWAY_URL}/search", files={"file": (uploaded.name, img_bytes, "image/jpeg")}, data={"x1": x1, "y1": y1, "x2": x2, "y2": y2, "search_label": selected_box.get("search_label","")}, timeout=60)
                    results = resp.json()
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    results = {}
            matches  = results.get("matches", [])
            det_cat  = results.get("detected_category", "?")
            mismatch = results.get("category_mismatch_warning")
            st.markdown(f"**Detected category:** `{det_cat}` · **{len(matches)} matches**")
            if mismatch:
                st.warning(f"⚠️ {mismatch}")
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
    test_title = st.text_input("Product title", placeholder="Adidas Wide Leg Track Pant Black")
    if st.button("Classify") and test_title.strip():
        with st.spinner("Classifying…"):
            try:
                resp = requests.post(f"{VISUAL_URL}/classify-text", json={"title": test_title}, timeout=10)
                data = resp.json()
                st.markdown(f"**Category:** `{data.get('category','None')}`")
                st.markdown(f"**Confidence:** `{data.get('confidence', 0):.4f}`")
            except Exception as e:
                st.error(f"Failed: {e}")

    st.divider()
    st.markdown("### Test Image Classification")
    tc1, tc2 = st.columns([1, 2])
    with tc1:
        test_img  = st.file_uploader("Image", type=["jpg","jpeg","png","webp"], key="test_img")
        test_name = st.text_input("Product title", placeholder="Adidas Hoodie Black", key="test_name")
        test_go   = st.button("Run analysis", type="primary")
    with tc2:
        if test_go and test_img and test_name:
            img_bytes = test_img.read()
            with st.spinner("Running…"):
                try:
                    resp = requests.post(f"{VISUAL_URL}/index-image", files={"file": (test_img.name, img_bytes, "image/jpeg")}, data={"title": test_name}, timeout=90)
                    data = resp.json()
                    if data.get("skipped"):
                        st.warning(f"⏭ Skipped: `{data.get('skip_reason')}`")
                    else:
                        cat   = data.get("category", "?")
                        color = CAT_COLORS.get(cat, "#888")
                        st.markdown(f'<span class="badge" style="background:{color}22;color:{color};border:1px solid {color}44;font-size:1rem;padding:6px 14px">{cat}</span>', unsafe_allow_html=True)
                        st.markdown(f"**Box source:** `{data.get('box_source','?')}`")
                        st.markdown(f"**Vector dims:** `{len(data.get('vector_normal',[]))}`")
                        st.success("Analysis complete (not saved to Qdrant)")
                except Exception as e:
                    st.error(f"Failed: {e}")

    st.divider()
    st.markdown("### Active Whitelist Overrides")
    if st.button("↻ Refresh overrides"):
        st.rerun()
    try:
        resp = requests.get(f"{VISUAL_URL}/whitelist-overrides", timeout=5)
        if resp.ok:
            entries = resp.json().get("entries", [])
            if entries:
                st.markdown(f"**{len(entries)} active overrides:**")
                cols = st.columns(3)
                for i, e in enumerate(entries):
                    with cols[i % 3]:
                        color = CAT_COLORS.get(e["category"], "#888")
                        st.markdown(f'`{e["word"]}` → <span class="badge" style="background:{color}22;color:{color};border:1px solid {color}44">{e["category"]}</span>', unsafe_allow_html=True)
            else:
                st.info("No active overrides yet.")
        else:
            st.warning("Could not reach visual engine.")
    except Exception as e:
        st.warning(f"Visual engine unreachable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — WHITELIST REVIEW
# Full-width layout:
#   Section 1: Pending suggestions (full width)
#   Section 2: Skipped products browser (full width, 3-column grid)
#   Section 3: Index health (full width)
# ══════════════════════════════════════════════════════════════════════════════
with tab_review:
    st.markdown("### Whitelist Review")
    st.markdown("Shop owners suggest keyword fixes for skipped products. Approving triggers an automatic re-index of matching products.")

    if st.button("↻ Refresh page", key="refresh_review"):
        st.rerun()

    # ── Section 1: Pending suggestions ───────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Pending suggestions")

    try:
        pending_resp = requests.get(f"{GATEWAY_URL}/whitelist-suggestions?status=pending", timeout=5)
        pending      = pending_resp.json().get("suggestions", []) if pending_resp.ok else []
    except Exception:
        pending = []
        st.error("Could not reach gateway.")

    if not pending:
        st.success("✓ No pending suggestions — nothing to review.")
    else:
        st.markdown(f"**{len(pending)} suggestion{'s' if len(pending) > 1 else ''} waiting:**")

        for entry in pending:
            word    = entry.get("word", "")
            cat     = entry.get("category", "")
            example = entry.get("example_product", "")
            store   = entry.get("store_name", "")
            created = entry.get("created_at", "")
            color   = CAT_COLORS.get(cat, "#888")

            st.markdown(
                f'<div class="pending-card">'
                f'<div style="font-size:1.1rem;font-weight:600;color:#f0ede8;margin-bottom:6px">'
                f'&ldquo;{word}&rdquo; &rarr; <span style="color:{color}">{cat}</span></div>'
                f'<div style="font-size:0.75rem;color:#6b6458;margin-bottom:4px">'
                f'Suggested by <strong style="color:#999">{store or "unknown"}</strong>'
                f'{" · " + created[:10] if created else ""}</div>'
                f'<div style="font-size:0.75rem;color:#555;font-style:italic">'
                f'Example: {example[:100] if example else "—"}</div></div>',
                unsafe_allow_html=True,
            )
            col_approve, col_reject, _ = st.columns([1, 1, 6])
            with col_approve:
                if st.button("✅ Approve", key=f"approve_{word}"):
                    with st.spinner(f"Approving '{word}'…"):
                        try:
                            resp = requests.post(f"{GATEWAY_URL}/whitelist-approve", json={"word": word}, timeout=15)
                            if resp.ok:
                                tokens_after = resp.json().get("visual_engine", {}).get("tokens_after", "?")
                                st.success(f"✅ Approved! Token map now has {tokens_after} tokens. Re-indexing in background.")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"Failed: {resp.text[:200]}")
                        except Exception as e:
                            st.error(f"Failed: {e}")
            with col_reject:
                if st.button("❌ Reject", key=f"reject_{word}"):
                    try:
                        resp = requests.post(f"{GATEWAY_URL}/whitelist-reject", json={"word": word}, timeout=5)
                        if resp.ok:
                            st.info(f"Rejected '{word}'.")
                            time.sleep(0.5)
                            st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")
            st.markdown("---")

    with st.expander("Previously reviewed (approved / rejected)"):
        try:
            all_resp        = requests.get(f"{GATEWAY_URL}/whitelist-suggestions", timeout=5)
            all_suggestions = all_resp.json().get("suggestions", []) if all_resp.ok else []
        except Exception:
            all_suggestions = []
        reviewed = [s for s in all_suggestions if s.get("status") in ("approved", "rejected")]
        if not reviewed:
            st.markdown("*No reviewed suggestions yet.*")
        else:
            for entry in reviewed:
                status  = entry.get("status", "")
                css_cls = "approved-card" if status == "approved" else "rejected-card"
                icon    = "✅" if status == "approved" else "❌"
                date    = entry.get("approved_at", entry.get("rejected_at", ""))[:10]
                st.markdown(
                    f'<div class="{css_cls}">{icon} <strong>{entry.get("word","")}</strong> → {entry.get("category","")} '
                    f'<span style="color:#555;font-size:0.7rem">({status}{" · " + date if date else ""})</span></div>',
                    unsafe_allow_html=True,
                )

    # ── Section 2: Skipped products browser (full width, 3-col grid) ─────────
    st.markdown("---")
    st.markdown("#### Skipped products")

    # Summary stats row
    try:
        skip_all_resp = requests.get(f"{GATEWAY_URL}/skipped-products?limit=500", timeout=5)
        skip_all_data = skip_all_resp.json() if skip_all_resp.ok else {}
    except Exception:
        skip_all_data = {}

    all_skipped_items = skip_all_data.get("products", [])
    total_skipped     = skip_all_data.get("total", 0)

    if total_skipped == 0:
        st.success("✓ No skipped products.")
    else:
        # Stats in a compact row
        reason_counts = {}
        store_counts  = {}
        for p in all_skipped_items:
            r = p.get("skip_reason", "unknown")
            s = p.get("store_name", "unknown")
            reason_counts[r] = reason_counts.get(r, 0) + 1
            store_counts[s]  = store_counts.get(s, 0) + 1

        stat_cols = st.columns(len(reason_counts) + 1)
        stat_cols[0].metric("Total skipped", total_skipped)
        for i, (reason, count) in enumerate(sorted(reason_counts.items(), key=lambda x: -x[1]), 1):
            stat_cols[i].metric(reason.replace("_", " "), count)

        # Store breakdown
        if store_counts:
            store_parts = " · ".join([f"**{s[:20]}**: {c}" for s, c in sorted(store_counts.items(), key=lambda x: -x[1])[:8]])
            st.markdown(f"<div style='font-size:0.78rem;color:#666;margin-bottom:12px'>By store: {store_parts}</div>", unsafe_allow_html=True)

    # Filter + pagination
    sf1, sf2 = st.columns([2, 6])
    skip_filter = sf1.selectbox("Filter", ["all", "no_consensus", "no_box_found"], key="skip_filter_main")

    if "skip_browse_offset" not in st.session_state:
        st.session_state.skip_browse_offset = 0
    if "skip_browse_last_filter" not in st.session_state:
        st.session_state.skip_browse_last_filter = skip_filter
    if skip_filter != st.session_state.skip_browse_last_filter:
        st.session_state.skip_browse_offset      = 0
        st.session_state.skip_browse_last_filter = skip_filter

    SKIP_PAGE = 12  # 3 columns × 4 rows

    try:
        params    = f"?limit={SKIP_PAGE}&offset={st.session_state.skip_browse_offset}"
        if skip_filter != "all":
            params += f"&skip_reason={skip_filter}"
        page_resp = requests.get(f"{GATEWAY_URL}/skipped-products{params}", timeout=5)
        page_data = page_resp.json() if page_resp.ok else {}
    except Exception:
        page_data = {}

    page_products = page_data.get("products", [])
    page_total    = page_data.get("total", 0)
    current_off   = st.session_state.skip_browse_offset

    if page_products:
        grid = st.columns(3)
        for i, p in enumerate(page_products):
            img_url = p.get("image_url", "")
            reason  = p.get("skip_reason", "")
            color   = REASON_COLORS.get(reason, "#888")

            with grid[i % 3]:
                with st.container():
                    c_img, c_info = st.columns([1, 2])
                    with c_img:
                        if img_url:
                            st.image(img_url, use_container_width=True)
                        else:
                            st.markdown("📦")
                    with c_info:
                        st.markdown(
                            f'<div style="font-size:0.75rem;color:#f0ede8;font-weight:500;'
                            f'line-height:1.4;margin-bottom:6px">{p.get("name","")[:60]}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<span style="font-size:0.65rem;color:{color};background:{color}18;'
                            f'border:1px solid {color}40;padding:2px 6px;border-radius:4px">'
                            f'{reason.replace("_"," ")}</span>',
                            unsafe_allow_html=True,
                        )
                        st.caption(p.get("store_name", ""))
                    st.markdown('<hr style="border:none;border-top:1px solid #1a1a1a;margin:8px 0">', unsafe_allow_html=True)

        # Pagination
        showing_from = current_off + 1
        showing_to   = min(current_off + SKIP_PAGE, page_total)
        nav1, nav2, nav3 = st.columns([1, 4, 1])
        if nav1.button("← Prev", key="skip_prev", disabled=current_off == 0):
            st.session_state.skip_browse_offset = max(0, current_off - SKIP_PAGE)
            st.rerun()
        nav2.markdown(f"<div style='text-align:center;color:#666;font-size:0.8rem;padding-top:8px'>{showing_from}–{showing_to} of {page_total}</div>", unsafe_allow_html=True)
        if nav3.button("Next →", key="skip_next", disabled=showing_to >= page_total):
            st.session_state.skip_browse_offset = current_off + SKIP_PAGE
            st.rerun()
    elif total_skipped > 0:
        st.markdown('<div style="color:#555;font-size:0.78rem">No products match this filter.</div>', unsafe_allow_html=True)

    # ── Section 3: Index health ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Index health")

    try:
        stats_resp = requests.get(f"{GATEWAY_URL}/index-stats", timeout=5)
        stats_data = stats_resp.json() if stats_resp.ok else {}
    except Exception:
        stats_data = {}

    if stats_data:
        stale = stats_data.get("by_tier", {}).get("full_image", 0) + stats_data.get("by_tier", {}).get("unknown", 0)
        if stale > 0:
            st.warning(f"⚠️ {stale} products have stale vectors. Run `reindex_stale.py`.")
        else:
            st.success("✓ All vectors are clean.")

        by_tier = stats_data.get("by_tier", {})
        total_p = stats_data.get("unique_products", 0)
        st.markdown(f"**{total_p} unique products indexed · {stats_data.get('total_points',0)} total points**")

        tier_cols = st.columns(len([k for k, v in by_tier.items() if v > 0]))
        col_idx   = 0
        for tier, count in by_tier.items():
            if count == 0:
                continue
            tier_color = {"exact": "#7aab8a", "alias": "#c9a96e", "best_available": "#c9c06e", "full_image": "#c97070", "unknown": "#c97070"}.get(tier, "#888")
            pct = round(count / total_p * 100) if total_p > 0 else 0
            tier_cols[col_idx].markdown(
                f'<div style="background:#111;border:1px solid #222;border-top:3px solid {tier_color};'
                f'border-radius:8px;padding:12px 16px;text-align:center">'
                f'<div style="color:{tier_color};font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px">{tier}</div>'
                f'<div style="color:#f0ede8;font-size:1.4rem;font-weight:600">{count}</div>'
                f'<div style="color:#555;font-size:0.75rem">{pct}%</div></div>',
                unsafe_allow_html=True,
            )
            col_idx += 1
    else:
        st.warning("Could not reach gateway for index stats.")