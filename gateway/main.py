import os
import uuid
import io
import json as _json
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from PIL import Image

app = FastAPI()

VISUAL_URL      = "http://visual_engine:8001"
QDRANT_HOST     = "qdrant"
QDRANT_PORT     = 6333
COLLECTION_NAME = "locus_items"

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


@app.on_event("startup")
def startup_event():
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
        print(f"Created Qdrant collection: {COLLECTION_NAME}")
    else:
        print(f"Qdrant collection exists: {COLLECTION_NAME}")


@app.get("/")
def read_root():
    return {"status": "online", "service": "Locus Gateway"}


# ── Feedback ───────────────────────────────────────────────────────────────────
@app.post("/feedback")
async def receive_feedback(
    query_category: str = Form("Unknown"),
    rating:         str = Form(...),
):
    if rating == "upvote":
        print(f"[FEEDBACK] SUCCESS: '{query_category}'")
    else:
        print(f"[FEEDBACK] FAILURE: '{query_category}'")
    return {"status": "logged"}


# ── Health Check ───────────────────────────────────────────────────────────────
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


# ── Step 1: Detect ─────────────────────────────────────────────────────────────
@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """Runs both YOLO detectors. Returns bounding boxes for the user to pick from."""
    async with httpx.AsyncClient() as http_client:
        files    = {"file": (file.filename, await file.read(), file.content_type)}
        response = await http_client.post(f"{VISUAL_URL}/detect", files=files, timeout=60.0)
        response.raise_for_status()
        return response.json()


# ── Step 2: Search ─────────────────────────────────────────────────────────────
@app.post("/search")
async def search(
    file:         UploadFile = File(...),
    x1:           int        = Form(None),
    y1:           int        = Form(None),
    x2:           int        = Form(None),
    y2:           int        = Form(None),
    search_label: str        = Form(None),   # for UI display only
):
    """
    Crops the selected item, vectorizes it, searches Qdrant.
    Category filter applied if CLIP is confident enough (≥ 0.45).
    Falls back to unfiltered search if no results in that category.
    """
    image_bytes = await file.read()
    is_cropped  = all(v is not None for v in [x1, y1, x2, y2])

    if is_cropped:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        filename    = "cropped_selection.png"
    else:
        filename = file.filename

    async with httpx.AsyncClient() as http_client:
        vis_response = await http_client.post(
            f"{VISUAL_URL}/vectorize",
            files={"file": (filename, image_bytes, "image/png")},
            data={"skip_rembg": str(is_cropped).lower()},
            timeout=60.0,
        )
        vis_response.raise_for_status()
        vis_data = vis_response.json()

    query_vector      = vis_data.get("vector")
    detected_category = vis_data.get("category")
    category_scores   = vis_data.get("category_scores", {})
    processed_image   = vis_data.get("debug_image")

    if not query_vector:
        raise HTTPException(status_code=500, detail="Visual engine returned no vector")

    # Category-filtered search first, fallback to unfiltered if empty
    if detected_category:
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=models.Filter(
                must=[models.FieldCondition(
                    key="category_tag",
                    match=models.MatchValue(value=detected_category)
                )]
            ),
            limit=50
        )
        if not search_result:
            print(f"No results for '{detected_category}' — retrying without filter")
            search_result = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=50
            )
    else:
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=50
        )

    matches = []
    for hit in search_result:
        matches.append({
            "name":      hit.payload.get("name", "Unknown"),
            "store":     hit.payload.get("store_name", "Unknown"),
            "mall":      hit.payload.get("mall_name", "Unknown"),
            "price":     hit.payload.get("price", ""),
            "score":     round(hit.score, 3),
            "image_url": hit.payload.get("filename"),
            "item_id":   hit.payload.get("item_id"),
            "category":  hit.payload.get("category_tag"),
        })

    return {
        "matches":           matches,
        "debug_image":       processed_image,
        "detected_category": detected_category,
        "category_scores":   category_scores,
    }


# ── Add Item (single, multipart) ───────────────────────────────────────────────
@app.post("/add")
async def add_item(
    name:  str        = Form(...),
    store: str        = Form(...),
    mall:  str        = Form(...),
    file:  UploadFile = File(...)
):
    """Single item upload via file. Used for manual testing."""
    async with httpx.AsyncClient() as http_client:
        vis_response = await http_client.post(
            f"{VISUAL_URL}/vectorize",
            files={"file": (file.filename, await file.read(), file.content_type)},
            timeout=60.0
        )
        vis_response.raise_for_status()
        data            = vis_response.json()
        vector          = data.get("vector")
        category_tag    = data.get("category")
        category_scores = data.get("category_scores", {})

    if not vector:
        raise HTTPException(status_code=500, detail="Visual engine returned no vector")

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(
            id      = str(uuid.uuid4()),
            vector  = vector,
            payload = {
                "name":            name,
                "store_name":      store,
                "mall_name":       mall,
                "filename":        file.filename,
                "category_tag":    category_tag,
                "category_scores": category_scores,
            }
        )]
    )
    return {"status": "indexed", "item": name, "category": category_tag}


# ══════════════════════════════════════════════════════════════════════════════
# /add-bulk — index a product from a public image URL
# ══════════════════════════════════════════════════════════════════════════════

class BulkItem(BaseModel):
    name:      str
    store:     str
    mall:      str
    image_url: str
    price:     str = ""


@app.post("/add-bulk")
async def add_bulk_item(item: BulkItem):
    """
    Index a single product from a JSON payload with a public image_url.

    Returns:
        { status: "indexed",  item, category }   — success
        { status: "rejected", item, reason   }   — CLIP not confident enough
    """
    async with httpx.AsyncClient() as http:

        # 1. Download image
        try:
            img_resp = await http.get(item.image_url, timeout=15.0, follow_redirects=True)
            img_resp.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot fetch image URL: {e}")

        content_type = img_resp.headers.get("content-type", "image/jpeg")

        # 2. Send to visual engine
        try:
            vis_resp = await http.post(
                f"{VISUAL_URL}/vectorize",
                files={"file": ("product.jpg", img_resp.content, content_type)},
                timeout=60.0,
            )
            vis_resp.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Visual engine error: {e}")

        vis_data        = vis_resp.json()
        vector          = vis_data.get("vector")
        category_tag    = vis_data.get("category")
        category_scores = vis_data.get("category_scores", {})

    if not vector:
        raise HTTPException(status_code=500, detail="Visual engine returned no vector")

    # 3. Reject if CLIP wasn't confident enough
    if category_tag is None:
        top_label = max(category_scores, key=category_scores.get) if category_scores else "unknown"
        top_score = category_scores.get(top_label, 0.0)
        return {
            "status": "rejected",
            "item":   item.name,
            "reason": f"CLIP confidence too low (best guess: '{top_label}' at {top_score:.0%}). "
                      f"Check that the image URL shows a single clear clothing item."
        }

    # 4. Upsert into Qdrant
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(
            id      = str(uuid.uuid4()),
            vector  = vector,
            payload = {
                "name":            item.name,
                "store_name":      item.store,
                "mall_name":       item.mall,
                "filename":        item.image_url,
                "category_tag":    category_tag,
                "category_scores": category_scores,
                "price":           item.price,
            }
        )]
    )
    return {"status": "indexed", "item": item.name, "category": category_tag}


# ══════════════════════════════════════════════════════════════════════════════
# /scrape — fetch a store URL, return product preview (no indexing)
# ══════════════════════════════════════════════════════════════════════════════

class ScrapeRequest(BaseModel):
    url:          str
    max_products: int = 20


@app.post("/scrape")
async def scrape_store(req: ScrapeRequest):
    """
    Scrapes a product listing page and returns a preview list.
    Three strategies: JSON-LD → Open Graph → HTML product cards.
    No indexing — dashboard confirms before calling /add-bulk.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as http:
        try:
            page_resp = await http.get(req.url, headers=headers)
            page_resp.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot reach URL: {e}")

    soup     = BeautifulSoup(page_resp.text, "lxml")
    products = []

    # Strategy 1: JSON-LD
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

    # Strategy 2: Open Graph
    if not products:
        og_image = soup.find("meta", property="og:image")
        og_title = soup.find("meta", property="og:title")
        og_price = soup.find("meta", property="product:price:amount")
        if og_image and og_image.get("content"):
            products.append({
                "name":      (og_title["content"] if og_title else "Product").strip(),
                "image_url": og_image["content"].strip(),
                "price":     (og_price["content"] if og_price else ""),
            })

    # Strategy 3: HTML product cards
    if not products:
        products = _scrape_product_cards(soup, req.url)

    # Deduplicate and cap
    seen, unique = set(), []
    for p in products:
        key = p.get("image_url", "")
        if key and key not in seen and p.get("name"):
            seen.add(key)
            unique.append(p)
        if len(unique) >= req.max_products:
            break

    return {"products": unique, "total_found": len(unique), "source_url": req.url}


# ── Scrape helpers ─────────────────────────────────────────────────────────────

def _extract_ld_product(obj: dict, products: list):
    name = obj.get("name", "").strip()
    if not name:
        return
    image = obj.get("image", "")
    if isinstance(image, list): image = image[0] if image else ""
    if isinstance(image, dict): image = image.get("url", "")
    price    = ""
    offers   = obj.get("offers", {})
    if isinstance(offers, list): offers = offers[0] if offers else {}
    if isinstance(offers, dict):
        price    = str(offers.get("price", ""))
        currency = offers.get("priceCurrency", "")
        if price and currency:
            price = f"{currency} {price}"
    if name and image:
        products.append({"name": name, "image_url": str(image).strip(), "price": price})


def _scrape_product_cards(soup: BeautifulSoup, base_url: str) -> list:
    from urllib.parse import urljoin
    products      = []
    card_keywords = [
        "product-card", "product-item", "product_card", "product_item",
        "item-card", "item_card", "grid-item", "catalogue-item",
        "collection-product", "product-grid-item",
    ]
    candidates = []
    for kw in card_keywords:
        candidates += soup.find_all(
            lambda tag, k=kw: tag.has_attr("class") and
            any(k in cls.lower() for cls in tag["class"])
        )
    seen_ids, unique_candidates = set(), []
    for c in candidates:
        eid = id(c)
        if eid not in seen_ids:
            seen_ids.add(eid)
            unique_candidates.append(c)
    for card in unique_candidates:
        img_tag = card.find("img")
        if not img_tag:
            continue
        img_url = img_tag.get("data-src") or img_tag.get("src") or ""
        if not img_url or img_url.startswith("data:"):
            continue
        img_url = urljoin(base_url, img_url)
        name_el = (
            card.find(class_=lambda c: c and "title" in c.lower()) or
            card.find(["h2", "h3", "h4"]) or
            img_tag
        )
        name  = name_el.get_text(strip=True) if hasattr(name_el, "get_text") else img_tag.get("alt", "").strip()
        if not name:
            name = "Product"
        price_tag = card.find(
            lambda t: t.has_attr("class") and
            any("price" in cls.lower() for cls in t["class"])
        )
        price = price_tag.get_text(strip=True) if price_tag else ""
        products.append({"name": name, "image_url": img_url, "price": price})
    return products