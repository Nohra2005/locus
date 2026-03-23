import asyncio
import io
import json as _json
import os
import uuid
from datetime import datetime
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, PointStruct, VectorParams

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VISUAL_URL           = os.getenv("VISUAL_HOST",    "http://visual_engine:8001")
QDRANT_URL           = os.getenv("QDRANT_URL")
QDRANT_API_KEY       = os.getenv("QDRANT_API_KEY")
QDRANT_HOST          = os.getenv("QDRANT_HOST",    "qdrant")
QDRANT_PORT          = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME      = "locus_items"
SKIPPED_COLLECTION   = "locus_skipped"        # separate collection for skipped products
PENDING_PATH         = "/app/pending_whitelist.json"

if QDRANT_URL:
    print(f"[QDRANT] Connecting to cloud: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    print(f"[QDRANT] Connecting to local: {QDRANT_HOST}:{QDRANT_PORT}")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


@app.on_event("startup")
def startup_event():
    # ── Main collection ───────────────────────────────────────────────────────
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
    for field in ("category_tag", "store_name", "product_id", "box_source"):
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema="keyword",
            )
        except Exception:
            pass

    # ── Skipped collection ────────────────────────────────────────────────────
    # Uses a dummy vector of size 1 — we never do vector search here,
    # only payload scroll and filter.
    if not client.collection_exists(collection_name=SKIPPED_COLLECTION):
        client.create_collection(
            collection_name=SKIPPED_COLLECTION,
            vectors_config=VectorParams(size=1, distance=Distance.COSINE),
        )
    for field in ("store_name", "skip_reason", "product_id"):
        try:
            client.create_payload_index(
                collection_name=SKIPPED_COLLECTION,
                field_name=field,
                field_schema="keyword",
            )
        except Exception:
            pass

    # ── Ensure pending whitelist file exists ──────────────────────────────────
    if not os.path.exists(PENDING_PATH):
        with open(PENDING_PATH, "w") as f:
            _json.dump([], f)


@app.get("/")
def read_root():
    return {"status": "online", "service": "Locus Gateway"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _crop_image_bytes(image_bytes: bytes, x1: float, y1: float, x2: float, y2: float) -> bytes:
    img  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W, H = img.size
    x1c = max(0, int(x1)); y1c = max(0, int(y1))
    x2c = min(W, int(x2)); y2c = min(H, int(y2))
    crop = img.crop((x1c, y1c, x2c, y2c))
    buf  = io.BytesIO()
    crop.save(buf, format="JPEG")
    return buf.getvalue()


def _read_pending() -> list:
    if not os.path.exists(PENDING_PATH):
        return []
    try:
        with open(PENDING_PATH, "r") as f:
            return _json.load(f)
    except Exception:
        return []


def _write_pending(data: list):
    with open(PENDING_PATH, "w") as f:
        _json.dump(data, f, indent=2)


def _store_skipped(
    product_id: str,
    name: str,
    image_url: str,
    store: str,
    mall: str,
    price: str,
    skip_reason: str,
):
    """Upsert a skipped product into locus_skipped collection."""
    try:
        client.upsert(
            collection_name=SKIPPED_COLLECTION,
            points=[
                PointStruct(
                    id     = str(uuid.uuid5(uuid.NAMESPACE_URL, f"skipped::{image_url or product_id}")),
                    vector = [0.0],   # dummy — never searched by vector
                    payload = {
                        "product_id":  product_id,
                        "name":        name,
                        "image_url":   image_url,
                        "store_name":  store,
                        "mall_name":   mall,
                        "price":       price,
                        "skip_reason": skip_reason,
                        "skipped_at":  datetime.utcnow().isoformat(),
                    }
                )
            ]
        )
    except Exception as e:
        print(f"[SKIPPED] Failed to store skipped product '{name}': {e}")


# ── Feedback ───────────────────────────────────────────────────────────────────

@app.post("/feedback")
async def receive_feedback(
    query_category: str = Form("Unknown"),
    rating:         str = Form(...),
):
    if rating == "upvote":
        print(f"[FEEDBACK] SUCCESS: '{query_category}'")
    else:
        print(f"[FEEDBACK] FAILURE: '{query_category}' — needs fine-tuning")
    return {"status": "logged"}


# ── Detect ─────────────────────────────────────────────────────────────────────

@app.post("/detect")
async def detect_items(file: UploadFile = File(...)):
    image_bytes = await file.read()
    async with httpx.AsyncClient() as http:
        resp = await http.post(
            f"{VISUAL_URL}/detect",
            files={"file": (file.filename, image_bytes, file.content_type)},
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    status = {"gateway": "ready", "visual_engine": "not_ready", "qdrant": "not_ready"}
    try:
        async with httpx.AsyncClient() as http_client:
            resp = await http_client.get(f"{VISUAL_URL}/", timeout=3.0)
            if resp.status_code == 200:
                status["visual_engine"] = "ready"
    except Exception:
        status["visual_engine"] = "loading"
    try:
        client.get_collections()
        status["qdrant"] = "ready"
    except Exception:
        status["qdrant"] = "loading"
    return {"ready": all(v == "ready" for v in status.values()), "services": status}


# ── Index Stats ────────────────────────────────────────────────────────────────

@app.get("/index-stats")
async def index_stats():
    try:
        source_counts: dict[str, int] = {}
        seen_product_ids: set         = set()
        total_points                  = 0
        offset                        = None

        while True:
            results, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=250,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not results:
                break
            for pt in results:
                total_points += 1
                p = pt.payload
                if p.get("is_dark") is True:
                    continue
                product_id = p.get("product_id", str(pt.id))
                if product_id in seen_product_ids:
                    continue
                seen_product_ids.add(product_id)
                src = p.get("box_source", "unknown")
                source_counts[src] = source_counts.get(src, 0) + 1
            if next_offset is None:
                break
            offset = next_offset

        tier_summary = {
            "exact": 0, "alias": 0, "best_available": 0,
            "full_image": 0, "unknown": 0, "other": 0,
        }
        for src, count in source_counts.items():
            if src.endswith("_exact"):            tier_summary["exact"] += count
            elif src.endswith("_alias"):          tier_summary["alias"] += count
            elif src.endswith("_best_available"): tier_summary["best_available"] += count
            elif src == "full_image":             tier_summary["full_image"] += count
            elif src == "unknown":                tier_summary["unknown"] += count
            else:                                 tier_summary["other"] += count

        return {
            "total_points":      total_points,
            "unique_products":   len(seen_product_ids),
            "by_box_source_raw": source_counts,
            "by_tier":           tier_summary,
            "reindex_required":  (tier_summary["full_image"] + tier_summary["unknown"]) > 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats query failed: {e}")


# ── Store Catalogue ────────────────────────────────────────────────────────────

@app.get("/store-catalogue")
async def store_catalogue(store_name: str, limit: int = 100, offset: int = 0):
    results, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(
                key="store_name",
                match=models.MatchValue(value=store_name)
            )]
        ),
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )
    seen_products = {}
    for point in results:
        p          = point.payload
        product_id = p.get("product_id", str(point.id))
        if product_id not in seen_products:
            seen_products[product_id] = {
                "id":           str(point.id),
                "name":         p.get("name", ""),
                "price":        p.get("price", ""),
                "category_tag": p.get("category_tag", ""),
                "image_url":    p.get("image_url", ""),
                "store_name":   p.get("store_name", ""),
                "mall_name":    p.get("mall_name", ""),
            }
    unique_products = list(seen_products.values())
    total           = len(unique_products)
    paginated       = unique_products[offset: offset + limit]
    return {"products": paginated, "total": total, "offset": offset, "limit": limit}


@app.delete("/store-catalogue/item/{item_id}")
async def delete_catalogue_item(item_id: str):
    try:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.PointIdsList(points=[item_id]),
        )
        return {"status": "deleted", "id": item_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Search ─────────────────────────────────────────────────────────────────────

@app.post("/search")
async def search_items(
    file:         UploadFile = File(...),
    x1:           float      = Form(0),
    y1:           float      = Form(0),
    x2:           float      = Form(0),
    y2:           float      = Form(0),
    search_label: str        = Form(""),
):
    image_bytes = await file.read()

    has_bbox = x2 > x1 and y2 > y1
    if has_bbox:
        crop_bytes = _crop_image_bytes(image_bytes, x1, y1, x2, y2)
        print(f"[SEARCH] Cropped to bbox ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
    else:
        crop_bytes = image_bytes
        print("[SEARCH] No bbox — using full image")

    async with httpx.AsyncClient() as http:
        vis_response = await http.post(
            f"{VISUAL_URL}/vectorize",
            files={"file": (file.filename, crop_bytes, "image/jpeg")},
            data={"yolo_label": search_label, "darken": "false"},
            timeout=60.0,
        )
        vis_response.raise_for_status()
        vis_data = vis_response.json()

    vector              = vis_data.get("vector")
    detected_category   = vis_data.get("category")
    category_confidence = vis_data.get("category_confidence", 0.0)
    processed_image     = vis_data.get("debug_image")

    # Category mismatch warning
    mismatch_warning = None
    if search_label and detected_category and search_label != detected_category:
        mismatch_warning = (
            f"Selected box category '{search_label}' differs from CLIP detection "
            f"'{detected_category}' (conf={category_confidence:.2f}). "
            f"Searching in '{search_label}' category as selected."
        )
        print(f"[SEARCH] WARNING: {mismatch_warning}")

    query_filter    = None
    effective_label = search_label or detected_category
    if effective_label:
        query_filter = models.Filter(
            must=[models.FieldCondition(
                key="category_tag",
                match=models.MatchValue(value=effective_label)
            )]
        )

    raw_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        query_filter=query_filter,
        limit=100,
    )

    best_per_product = {}
    for hit in raw_results:
        product_id = hit.payload.get("product_id", hit.payload.get("image_url", str(hit.id)))
        if product_id not in best_per_product or hit.score > best_per_product[product_id]["score"]:
            best_per_product[product_id] = {
                "name":       hit.payload.get("name", "Unknown"),
                "store_name": hit.payload.get("store_name", "Unknown"),
                "mall_name":  hit.payload.get("mall_name", "Unknown"),
                "price":      hit.payload.get("price", ""),
                "score":      round(hit.score, 3),
                "image_url":  hit.payload.get("image_url", ""),
                "item_id":    hit.payload.get("item_id", ""),
            }

    matches = sorted(best_per_product.values(), key=lambda x: x["score"], reverse=True)[:25]

    return {
        "matches":                   matches,
        "debug_image":               processed_image,
        "detected_category":         detected_category,
        "category_confidence":       category_confidence,
        "category_mismatch_warning": mismatch_warning,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SKIPPED PRODUCTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/skipped-products")
async def get_skipped_products(
    store_name:  str = "",
    skip_reason: str = "",
    limit:       int = 50,
    offset:      int = 0,
):
    must_conditions = []
    if store_name:
        must_conditions.append(models.FieldCondition(
            key="store_name", match=models.MatchValue(value=store_name)
        ))
    if skip_reason:
        must_conditions.append(models.FieldCondition(
            key="skip_reason", match=models.MatchValue(value=skip_reason)
        ))

    scroll_filter = models.Filter(must=must_conditions) if must_conditions else None

    # Fetch ALL matching results then paginate in Python.
    # Qdrant scroll offset is a point ID cursor, not a numeric offset —
    # passing offset=8 skips to point ID 8, not the 8th result.
    all_results = []
    cursor      = None

    while True:
        batch, next_cursor = client.scroll(
            collection_name=SKIPPED_COLLECTION,
            scroll_filter=scroll_filter,
            limit=250,
            offset=cursor,
            with_payload=True,
            with_vectors=False,
        )
        if not batch:
            break
        all_results.extend(batch)
        if next_cursor is None:
            break
        cursor = next_cursor

    total    = len(all_results)
    paginated = all_results[offset: offset + limit]
    products  = [pt.payload | {"point_id": str(pt.id)} for pt in paginated]

    return {
        "products": products,
        "total":    total,
        "offset":   offset,
        "limit":    limit,
    }

@app.delete("/skipped-products/{point_id}")
async def delete_skipped_product(point_id: str):
    """Remove a product from the skipped list (e.g. after manually resolving)."""
    try:
        client.delete(
            collection_name=SKIPPED_COLLECTION,
            points_selector=models.PointIdsList(points=[point_id]),
        )
        return {"status": "deleted", "id": point_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# WHITELIST SUGGESTIONS
# ══════════════════════════════════════════════════════════════════════════════

class WhitelistSuggestRequest(BaseModel):
    word:            str
    category:        str
    example_product: str = ""
    store_name:      str = ""


class WhitelistDecisionRequest(BaseModel):
    word: str


@app.post("/whitelist-suggest")
async def whitelist_suggest(req: WhitelistSuggestRequest):
    """
    Shop owner or developer suggests a new word → category mapping.
    Written to pending_whitelist.json with status 'pending'.
    """
    word     = req.word.strip().lower()
    category = req.category.strip()

    if not word or not category:
        raise HTTPException(status_code=400, detail="word and category are required")

    pending = _read_pending()

    # Don't add duplicates — update if already pending
    existing = next((e for e in pending if e.get("word") == word), None)
    if existing:
        existing["category"]        = category
        existing["example_product"] = req.example_product
        existing["store_name"]      = req.store_name
        existing["status"]          = "pending"
        existing["updated_at"]      = datetime.utcnow().isoformat()
    else:
        pending.append({
            "word":            word,
            "category":        category,
            "example_product": req.example_product,
            "store_name":      req.store_name,
            "status":          "pending",
            "created_at":      datetime.utcnow().isoformat(),
        })

    _write_pending(pending)
    print(f"[WHITELIST] Suggestion added: '{word}' → '{category}'")
    return {"status": "suggested", "word": word, "category": category}


@app.get("/whitelist-suggestions")
async def whitelist_suggestions(status: str = ""):
    """
    List whitelist suggestions. Filter by status: pending | approved | rejected.
    """
    pending = _read_pending()
    if status:
        pending = [e for e in pending if e.get("status") == status]
    return {"total": len(pending), "suggestions": pending}


@app.post("/whitelist-approve")
async def whitelist_approve(req: WhitelistDecisionRequest):
    """
    Approve a pending suggestion:
      1. Updates status in pending_whitelist.json
      2. Calls /whitelist-add on visual engine → writes override + reloads token map
      3. Triggers background re-index of matching skipped products
    """
    word    = req.word.strip().lower()
    pending = _read_pending()

    entry = next((e for e in pending if e.get("word") == word), None)
    if not entry:
        raise HTTPException(status_code=404, detail=f"No suggestion found for word '{word}'")

    category = entry["category"]

    # Step 1: update status
    entry["status"]      = "approved"
    entry["approved_at"] = datetime.utcnow().isoformat()
    _write_pending(pending)

    # Step 2: push to visual engine
    try:
        async with httpx.AsyncClient() as http:
            resp = await http.post(
                f"{VISUAL_URL}/whitelist-add",
                json={"word": word, "category": category},
                timeout=10.0,
            )
            resp.raise_for_status()
            ve_result = resp.json()
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Visual engine whitelist-add failed: {e}"
        )

    # Step 3: trigger background re-index of matching skipped products
    asyncio.create_task(_reindex_matching_skipped(word, category))

    return {
        "status":         "approved",
        "word":           word,
        "category":       category,
        "visual_engine":  ve_result,
        "reindex":        "triggered in background",
    }


@app.post("/whitelist-reject")
async def whitelist_reject(req: WhitelistDecisionRequest):
    """Reject a pending suggestion."""
    word    = req.word.strip().lower()
    pending = _read_pending()

    entry = next((e for e in pending if e.get("word") == word), None)
    if not entry:
        raise HTTPException(status_code=404, detail=f"No suggestion found for word '{word}'")

    entry["status"]      = "rejected"
    entry["rejected_at"] = datetime.utcnow().isoformat()
    _write_pending(pending)

    return {"status": "rejected", "word": word}


# ── Background: re-index skipped products matching an approved word ────────────

async def _reindex_matching_skipped(word: str, category: str):
    """
    Background task. After a word is approved, find all skipped products
    whose name contains that word (case-insensitive), re-index them,
    and if successful move them to locus_items + remove from locus_skipped.
    """
    print(f"[REINDEX] Starting background re-index for word='{word}' category='{category}'")

    # Scroll all skipped products — filter in Python (word match is text search)
    all_skipped = []
    offset = None
    while True:
        results, next_offset = client.scroll(
            collection_name=SKIPPED_COLLECTION,
            limit=250,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            break
        for pt in results:
            p = pt.payload
            if word.lower() in p.get("name", "").lower():
                all_skipped.append({"point_id": str(pt.id), **p})
        if next_offset is None:
            break
        offset = next_offset

    if not all_skipped:
        print(f"[REINDEX] No skipped products match word='{word}'")
        return

    print(f"[REINDEX] Found {len(all_skipped)} skipped products matching '{word}'")

    semaphore = asyncio.Semaphore(3)

    async def reindex_one(product: dict):
        async with semaphore:
            name      = product.get("name", "")
            img_url   = product.get("image_url", "")
            point_id  = product.get("point_id", "")
            store     = product.get("store_name", "")
            mall      = product.get("mall_name", "")
            price     = product.get("price", "")
            product_id = product.get("product_id", "")

            if not img_url:
                return

            try:
                async with httpx.AsyncClient() as http:
                    img_resp = await http.get(img_url, timeout=15.0, follow_redirects=True)
                    img_resp.raise_for_status()
                    img_bytes    = img_resp.content
                    content_type = img_resp.headers.get("content-type", "image/jpeg")

                    idx_resp = await http.post(
                        f"{VISUAL_URL}/index-image",
                        files={"file": ("product.jpg", img_bytes, content_type)},
                        data={"title": name},
                        timeout=90.0,
                    )
                    idx_resp.raise_for_status()
                    idx_data = idx_resp.json()

                if idx_data.get("skipped"):
                    print(f"  [REINDEX] Still skipped: '{name}' — {idx_data.get('skip_reason')}")
                    return

                # Successfully indexed — upsert to locus_items
                vector_normal  = idx_data["vector_normal"]
                final_category = idx_data.get("category", category)
                new_box_source = idx_data.get("box_source", "unknown")

                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[
                        PointStruct(
                            id      = str(uuid.uuid5(uuid.NAMESPACE_URL, img_url)),
                            vector  = vector_normal,
                            payload = {
                                "name":         name,
                                "store_name":   store,
                                "mall_name":    mall,
                                "image_url":    img_url,
                                "category_tag": final_category,
                                "price":        price,
                                "product_id":   product_id,
                                "box_source":   new_box_source,
                            }
                        )
                    ]
                )

                # Remove from skipped collection
                if point_id:
                    client.delete(
                        collection_name=SKIPPED_COLLECTION,
                        points_selector=models.PointIdsList(points=[point_id]),
                    )

                print(f"  [REINDEX] OK: '{name}' → {final_category} ({new_box_source})")

            except Exception as e:
                print(f"  [REINDEX] Failed: '{name}' — {e}")

    await asyncio.gather(*[reindex_one(p) for p in all_skipped])
    print(f"[REINDEX] Done for word='{word}'")


# ══════════════════════════════════════════════════════════════════════════════
# /add-bulk-batch
# ══════════════════════════════════════════════════════════════════════════════

class BulkBatchRequest(BaseModel):
    items: list[dict]


@app.post("/add-bulk-batch")
async def add_bulk_batch(batch: BulkBatchRequest):
    semaphore = asyncio.Semaphore(5)

    async def index_one(raw: dict):
        name   = raw.get("name", "Product")
        store  = raw.get("store", "")
        mall   = raw.get("mall", "")
        price  = raw.get("price", "")

        image_urls = raw.get("image_urls") or []
        if not image_urls and raw.get("image_url"):
            image_urls = [raw["image_url"]]
        if not image_urls:
            return {"status": "failed", "item": name, "error": "no image URL"}

        img_url    = image_urls[0]
        product_id = raw.get("product_id") or str(uuid.uuid5(
            uuid.NAMESPACE_URL, f"{name}::{store}"
        ))

        async with semaphore:
            try:
                async with httpx.AsyncClient() as http:

                    img_resp = await http.get(img_url, timeout=15.0, follow_redirects=True)
                    img_resp.raise_for_status()
                    img_bytes    = img_resp.content
                    content_type = img_resp.headers.get("content-type", "image/jpeg")

                    idx_resp = await http.post(
                        f"{VISUAL_URL}/index-image",
                        files={"file": ("product.jpg", img_bytes, content_type)},
                        data={"title": name},
                        timeout=90.0,
                    )
                    idx_resp.raise_for_status()
                    idx_data = idx_resp.json()

                    if idx_data.get("skipped"):
                        reason = idx_data.get("skip_reason", "unknown")
                        print(f"[BATCH] Skipped '{name}': {reason}")

                        # Store in locus_skipped so shop owner can review
                        _store_skipped(
                            product_id  = product_id,
                            name        = name,
                            image_url   = img_url,
                            store       = store,
                            mall        = mall,
                            price       = price,
                            skip_reason = reason,
                        )

                        return {"status": "skipped", "item": name, "reason": reason}

                    vector_normal  = idx_data["vector_normal"]
                    final_category = idx_data.get("category", "unknown")
                    box_source     = idx_data.get("box_source", "unknown")

                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=[
                            PointStruct(
                                id      = str(uuid.uuid5(uuid.NAMESPACE_URL, img_url)),
                                vector  = vector_normal,
                                payload = {
                                    "name":         name,
                                    "store_name":   store,
                                    "mall_name":    mall,
                                    "image_url":    img_url,
                                    "category_tag": final_category,
                                    "price":        price,
                                    "product_id":   product_id,
                                    "box_source":   box_source,
                                }
                            ),
                        ]
                    )

                return {"status": "ok", "item": name}

            except Exception as e:
                print(f"[BATCH] Failed: {name} — {e}")
                return {"status": "failed", "item": name, "error": str(e)}

    results = await asyncio.gather(*[index_one(raw) for raw in batch.items])
    success = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed  = [r for r in results if r["status"] == "failed"]

    return {
        "success": success,
        "skipped": skipped,
        "total":   len(batch.items),
        "failed":  failed,
    }


# ══════════════════════════════════════════════════════════════════════════════
# /scrape
# ══════════════════════════════════════════════════════════════════════════════

class ScrapeRequest(BaseModel):
    url:          str
    max_products: int = 0


@app.post("/scrape")
async def scrape_store(req: ScrapeRequest):
    req_headers = {
        "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control":   "no-cache",
        "Referer":         "https://www.google.com/",
    }

    parsed   = urlparse(req.url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"

    async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as http:
        shopify_products = await _try_shopify_api(
            http, req.url, base_url, req.max_products, req_headers
        )
        if shopify_products:
            return {
                "products":    shopify_products,
                "total_found": len(shopify_products),
                "source_url":  req.url,
                "strategy":    "shopify_api",
            }

        try:
            page_resp = await http.get(req.url, headers=req_headers)
            if page_resp.status_code == 429:
                return {
                    "products":    [],
                    "total_found": 0,
                    "source_url":  req.url,
                    "strategy":    "rate_limited",
                    "error":       "Site is rate-limiting. Use the Chrome extension scraper instead.",
                }
            if page_resp.status_code not in (200, 301, 302):
                raise HTTPException(
                    status_code=400,
                    detail=f"Site returned HTTP {page_resp.status_code}"
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot reach URL: {e}")

        soup     = BeautifulSoup(page_resp.text, "lxml")
        products = []

        for script_tag in soup.find_all("script", type="application/ld+json"):
            try:
                data  = _json.loads(script_tag.string or "")
                items = data if isinstance(data, list) else [data]
                for obj in items:
                    if obj.get("@type") == "ItemList":
                        for element in obj.get("itemListElement", []):
                            _extract_ld_product(element.get("item", element), products)
                    elif obj.get("@type") in ("Product", "product"):
                        _extract_ld_product(obj, products)
                    elif "@graph" in obj:
                        for node in obj["@graph"]:
                            if node.get("@type") == "Product":
                                _extract_ld_product(node, products)
            except Exception:
                continue

        if not products:
            og_image = soup.find("meta", property="og:image")
            og_title = soup.find("meta", property="og:title")
            og_price = soup.find("meta", property="product:price:amount")
            if og_image and og_image.get("content"):
                products.append({
                    "name":       (og_title["content"] if og_title else "Product").strip(),
                    "image_url":  og_image["content"].strip(),
                    "image_urls": [og_image["content"].strip()],
                    "price":      (og_price["content"] if og_price else ""),
                })

        if not products:
            products = _scrape_product_cards(soup, req.url)

    seen, unique = set(), []
    for p in products:
        key = p.get("image_url", "") or (p.get("image_urls") or [""])[0]
        if key and key not in seen and p.get("name"):
            seen.add(key)
            unique.append(p)

    return {
        "products":    unique,
        "total_found": len(unique),
        "source_url":  req.url,
        "strategy":    "html_fallback",
    }


async def _try_shopify_api(http, original_url, base_url, max_products, headers):
    path  = urlparse(original_url).path
    parts = [p for p in path.split("/") if p]

    base_endpoints = []
    if len(parts) >= 2 and parts[0] == "collections":
        base_endpoints.append(f"{base_url}/collections/{parts[1]}/products.json")
    base_endpoints.append(f"{base_url}/products.json")

    for base_endpoint in base_endpoints:
        try:
            all_products = []
            page         = 1

            while True:
                api_url = f"{base_endpoint}?limit=250&page={page}"
                resp    = await http.get(api_url, headers=headers, timeout=15.0)

                print(f"[SCRAPE] {api_url} → {resp.status_code} ({len(resp.content)} bytes)")

                if resp.status_code != 200:
                    break

                try:
                    import gzip as _gzip
                    content = resp.content
                    if content[:2] == b'\x1f\x8b':
                        content = _gzip.decompress(content)
                    data = _json.loads(content)
                except Exception:
                    print(f"[SCRAPE] Could not parse JSON: {resp.content[:50]}")
                    break

                raw_products = data.get("products", [])
                if not raw_products:
                    break

                for p in raw_products:
                    name = p.get("title", "").strip()
                    if not name:
                        continue
                    images     = p.get("images", [])
                    image_urls = [img.get("src", "") for img in images if img.get("src")]
                    image_url  = image_urls[0] if image_urls else ""
                    if not image_url:
                        continue
                    price    = ""
                    variants = p.get("variants", [])
                    if variants:
                        pv = variants[0].get("price", "")
                        if pv:
                            price = f"USD {pv}"
                    all_products.append({
                        "name":       name,
                        "image_url":  image_url,
                        "image_urls": image_urls,
                        "price":      price,
                    })

                if max_products and len(all_products) >= max_products:
                    all_products = all_products[:max_products]
                    break
                if len(raw_products) < 250:
                    break
                page += 1

            if all_products:
                return all_products

        except Exception as e:
            print(f"[SCRAPE] Shopify API failed: {e}")
            continue

    return []


def _extract_ld_product(obj: dict, products: list):
    name = obj.get("name", "").strip()
    if not name:
        return
    image = obj.get("image", "")
    if isinstance(image, list): image = image[0] if image else ""
    if isinstance(image, dict): image = image.get("url", "")
    price  = ""
    offers = obj.get("offers", {})
    if isinstance(offers, list): offers = offers[0] if offers else {}
    if isinstance(offers, dict):
        pv = str(offers.get("price", ""))
        cu = offers.get("priceCurrency", "")
        if pv and cu:
            price = f"{cu} {pv}"
    if name and image:
        s = str(image).strip()
        products.append({"name": name, "image_url": s, "image_urls": [s], "price": price})


def _scrape_product_cards(soup: BeautifulSoup, base_url: str) -> list:
    products = []
    for card in soup.select("[class*='product'], [class*='item'], article")[:50]:
        img = card.find("img")
        if not img:
            continue
        src = img.get("src") or img.get("data-src", "")
        if not src or src.startswith("data:"):
            continue
        if not src.startswith("http"):
            src = base_url.rstrip("/") + "/" + src.lstrip("/")
        title_el = card.find(["h2", "h3", "h4", "a"])
        name     = title_el.get_text(strip=True) if title_el else "Product"
        if len(name) > 120:
            continue
        products.append({"name": name, "image_url": src, "image_urls": [src], "price": ""})
    return products