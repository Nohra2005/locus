import os
import uuid
import io
import json as _json
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from PIL import Image
from urllib.parse import urlparse

app = FastAPI()

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASET_DIR = os.environ.get("DATASET_DIR", "/app/dataset")
try:
    if os.path.exists(DATASET_DIR):
        app.mount("/static", StaticFiles(directory=DATASET_DIR), name="static")
        print(f"Serving dataset images from {DATASET_DIR}")
    else:
        print(f"WARNING: DATASET_DIR not found: {DATASET_DIR}")
except Exception as e:
    print(f"WARNING: Could not mount static files: {e}")

VISUAL_URL      = "http://visual_engine:8001"
QDRANT_HOST     = "qdrant"
QDRANT_PORT     = 6333
COLLECTION_NAME = "locus_items"

LABEL_TO_CATEGORY = {
    "short sleeved shirt":   "blouses shirts",
    "long sleeved shirt":    "blouses shirts",
    "short sleeved outwear": "jackets vests",
    "long sleeved outwear":  "jackets vests",
    "vest":                  "jackets vests",
    "sling":                 "blouses shirts",
    "shorts":                "shorts",
    "trousers":              "pants capris",
    "skirt":                 "skirts",
    "short sleeved dress":   "dresses",
    "long sleeved dress":    "dresses",
    "vest dress":            "dresses",
    "sling dress":           "dresses",
    "shoe":                  "shoes",
    "bag, wallet":           "accessories",
    "glasses":               "accessories",
    "hat":                   "accessories",
    "shirt":                 "blouses shirts",
    "t-shirt":               "blouses shirts",
    "dress":                 "dresses",
    "pants":                 "pants capris",
    "jeans":                 "denim",
    "jacket":                "jackets vests",
    "coat":                  "jackets vests",
    "shoes":                 "shoes",
    "sneakers":              "shoes",
    "bag":                   "accessories",
    "handbag":               "accessories",
}

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

@app.on_event("startup")
def startup_event():
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )

@app.get("/")
def read_root():
    return {"status": "online", "service": "Locus Gateway"}


# ── Feedback ───────────────────────────────────────────────────────────────────
@app.post("/feedback")
async def receive_feedback(
    query_category: str = Form("Unknown"),
    rating: str = Form(...),
):
    if rating == "upvote":
        print(f"[FEEDBACK] SUCCESS: User loved results for '{query_category}'")
    else:
        print(f"[FEEDBACK] FAILURE: User rejected '{query_category}'. Needs fine-tuning.")
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
    async with httpx.AsyncClient() as http_client:
        files    = {"file": (file.filename, await file.read(), file.content_type)}
        response = await http_client.post(f"{VISUAL_URL}/detect", files=files, timeout=60.0)
        response.raise_for_status()
        return response.json()


# ── Step 2: Search ─────────────────────────────────────────────────────────────
@app.post("/search")
async def search(
    file: UploadFile = File(...),
    x1: int = Form(None),
    y1: int = Form(None),
    x2: int = Form(None),
    y2: int = Form(None),
    search_label: str = Form(None),
):
    image_bytes = await file.read()
    is_cropped  = all(v is not None for v in [x1, y1, x2, y2])

    if is_cropped:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

    async with httpx.AsyncClient() as http_client:
        vis_response = await http_client.post(
            f"{VISUAL_URL}/vectorize",
            files={"file": ("crop.png", image_bytes, "image/png")},
            data={"skip_rembg": "true" if is_cropped else "false",
                  "yolo_label": search_label or ""},
            timeout=60.0,
        )
        vis_response.raise_for_status()
        vis_data = vis_response.json()

    vector              = vis_data.get("vector")
    detected_category   = vis_data.get("category")
    category_confidence = vis_data.get("category_confidence", 0.0)
    processed_image     = vis_data.get("debug_image")

    query_filter = None
    effective_label = search_label or detected_category
    if effective_label:
        query_filter = models.Filter(
            must=[models.FieldCondition(
                key="category_tag",
                match=models.MatchValue(value=effective_label)
            )]
        )

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        query_filter=query_filter,
        limit=20,
    )

    matches = []
    for hit in results:
        matches.append({
            "name":           hit.payload.get("name", "Unknown"),
            "store_name":     hit.payload.get("store_name", "Unknown"),
            "mall_name":      hit.payload.get("mall_name", "Unknown"),
            "price":          hit.payload.get("price", "Unknown"),
            "score":          round(hit.score, 3),
            "image_filename": hit.payload.get("filename"),
            "item_id":        hit.payload.get("item_id"),
        })

    return {
        "matches":             matches,
        "debug_image":         processed_image,
        "detected_category":   detected_category,
        "category_confidence": category_confidence
    }


# ── Add Item (single, multipart form) ─────────────────────────────────────────
@app.post("/add")
async def add_item(
    name:  str        = Form(...),
    store: str        = Form(...),
    mall:  str        = Form(...),
    file:  UploadFile = File(...)
):
    async with httpx.AsyncClient() as http_client:
        vis_response = await http_client.post(
            f"{VISUAL_URL}/vectorize",
            files={"file": (file.filename, await file.read(), file.content_type)},
            timeout=30.0
        )
        vis_response.raise_for_status()
        data              = vis_response.json()
        vector            = data.get("vector")
        detected_category = data.get("category")

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(
            id      = str(uuid.uuid4()),
            vector  = vector,
            payload = {
                "name":         name,
                "store_name":   store,
                "mall_name":    mall,
                "filename":     file.filename,
                "category_tag": detected_category
            }
        )]
    )
    return {"status": "saved", "item": name}


# ══════════════════════════════════════════════════════════════════════════════
# /add-bulk  — index a product from a public image URL (JSON body)
# ══════════════════════════════════════════════════════════════════════════════

class BulkItem(BaseModel):
    name:      str
    store:     str
    mall:      str
    image_url: str
    price:     str = ""
    category:  str = ""


@app.post("/add-bulk")
async def add_bulk_item(item: BulkItem):
    async with httpx.AsyncClient() as http:
        try:
            img_resp = await http.get(item.image_url, timeout=15.0, follow_redirects=True)
            img_resp.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot fetch image URL: {e}")

        content_type = img_resp.headers.get("content-type", "image/jpeg")

        vis_resp = await http.post(
            f"{VISUAL_URL}/vectorize",
            files={"file": ("product.jpg", img_resp.content, content_type)},
            timeout=60.0,
        )
        vis_resp.raise_for_status()
        vis_data          = vis_resp.json()
        vector            = vis_data.get("vector")
        detected_category = vis_data.get("category")

    final_category = detected_category or item.category or "unknown"

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(
            id      = str(uuid.uuid4()),
            vector  = vector,
            payload = {
                "name":         item.name,
                "store_name":   item.store,
                "mall_name":    item.mall,
                "filename":     item.image_url,
                "category_tag": final_category,
                "price":        item.price,
            }
        )]
    )

    return {"status": "indexed", "item": item.name, "category": final_category}


# ══════════════════════════════════════════════════════════════════════════════
# /scrape  — smart scraper with Shopify API as Strategy 1
#
# Strategy order:
#   1. Shopify products.json API  (no auth needed, works on all Shopify stores)
#   2. JSON-LD structured data    (WooCommerce, most modern stores)
#   3. Open Graph meta tags       (single-product pages)
#   4. Generic product card HTML  (last resort)
# ══════════════════════════════════════════════════════════════════════════════

class ScrapeRequest(BaseModel):
    url:          str
    max_products: int = 0   # 0 = no limit — fetches all pages automatically


@app.post("/scrape")
async def scrape_store(req: ScrapeRequest):
    req_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    parsed   = urlparse(req.url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"

    async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as http:

        # ── Strategy 1: Shopify products.json API ─────────────────────────────
        shopify_products = await _try_shopify_api(
            http, req.url, base_url, req.max_products, req_headers
        )
        if shopify_products:
            print(f"[SCRAPE] Shopify API: found {len(shopify_products)} products")
            return {
                "products":    shopify_products,
                "total_found": len(shopify_products),
                "source_url":  req.url,
                "strategy":    "shopify_api",
            }

        # ── Strategies 2–4: HTML fallback ─────────────────────────────────────
        try:
            page_resp = await http.get(req.url, headers=req_headers)
            page_resp.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot reach URL: {e}")

        soup     = BeautifulSoup(page_resp.text, "lxml")
        products = []

        # Strategy 2: JSON-LD
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

        # Strategy 3: Open Graph
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

        # Strategy 4: HTML cards
        if not products:
            products = _scrape_product_cards(soup, req.url)

    # Deduplicate and cap
    seen, unique = set(), []
    for p in products:
        key = p.get("image_url", "")
        if key and key not in seen and p.get("name"):
            seen.add(key)
            unique.append(p)

    return {
        "products":    unique,
        "total_found": len(unique),
        "source_url":  req.url,
        "strategy":    "html_fallback",
    }


# ── Shopify API helper ─────────────────────────────────────────────────────────

async def _try_shopify_api(
    http: httpx.AsyncClient,
    original_url: str,
    base_url: str,
    max_products: int,
    headers: dict,
) -> list:
    """
    Tries Shopify's public /products.json endpoint (no API key needed).
    Paginates automatically — fetches all products, 250 per page, until
    the store returns an empty page (meaning we have everything).

    Two base endpoints tried:
      A) /collections/{handle}/products.json  — if URL targets a collection
      B) /products.json                        — store-wide fallback
    """
    path  = urlparse(original_url).path
    parts = [p for p in path.split("/") if p]

    base_endpoints = []
    if len(parts) >= 2 and parts[0] == "collections":
        collection_handle = parts[1]
        base_endpoints.append(
            f"{base_url}/collections/{collection_handle}/products.json"
        )
    base_endpoints.append(f"{base_url}/products.json")

    for base_endpoint in base_endpoints:
        try:
            all_products = []
            page         = 1

            while True:
                api_url = f"{base_endpoint}?limit=250&page={page}"
                print(f"[SCRAPE] Shopify API page {page}: {api_url}")

                resp = await http.get(api_url, headers=headers, timeout=15.0)
                if resp.status_code != 200:
                    break

                data         = resp.json()
                raw_products = data.get("products", [])

                if not raw_products:
                    # Empty page — we've fetched everything
                    break

                for p in raw_products:
                    name = p.get("title", "").strip()
                    if not name:
                        continue

                    image_url = ""
                    images = p.get("images", [])
                    if images:
                        image_url = images[0].get("src", "")

                    price = ""
                    variants = p.get("variants", [])
                    if variants:
                        price_val = variants[0].get("price", "")
                        if price_val:
                            price = f"USD {price_val}"

                    if name and image_url:
                        all_products.append({
                            "name":      name,
                            "image_url": image_url,
                            "price":     price,
                        })

                print(f"[SCRAPE] Page {page}: got {len(raw_products)} products (total so far: {len(all_products)})")

                # If we got fewer than 250, this was the last page
                if len(raw_products) < 250:
                    break

                page += 1

            if all_products:
                print(f"[SCRAPE] Shopify API complete: {len(all_products)} total products")
                return all_products

        except Exception as e:
            print(f"[SCRAPE] Shopify API failed ({base_endpoint}): {e}")
            continue

    return []


# ── HTML scrape helpers ────────────────────────────────────────────────────────

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
        name = name_el.get_text(strip=True) if hasattr(name_el, "get_text") else img_tag.get("alt", "").strip()
        if not name:
            name = "Product"
        price_tag = card.find(
            lambda t: t.has_attr("class") and
            any("price" in cls.lower() for cls in t["class"])
        )
        price = price_tag.get_text(strip=True) if price_tag else ""
        products.append({"name": name, "image_url": img_url, "price": price})
    return products