import asyncio
import json as _json
import os
import uuid
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VISUAL_URL       = os.getenv("VISUAL_HOST",   "http://visual_engine:8001")
RANKING_URL      = os.getenv("RANKING_HOST",  "http://ranking_engine:8002")
QDRANT_HOST      = os.getenv("QDRANT_HOST",   "qdrant")
QDRANT_PORT      = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME  = "locus_items"

client = QdrantClient(url=QDRANT_HOST, port=QDRANT_PORT)

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
    rating:         str = Form(...),
):
    if rating == "upvote":
        print(f"[FEEDBACK] SUCCESS: User loved results for '{query_category}'")
    else:
        print(f"[FEEDBACK] FAILURE: User rejected '{query_category}'. Needs fine-tuning.")
    return {"status": "logged"}

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


# ── Store Catalogue ────────────────────────────────────────────────────────────
@app.get("/store-catalogue")
async def store_catalogue(store_name: str, limit: int = 100, offset: int = 0):
    """
    Returns unique products for a given store_name.
    Deduplicates by product_id so multi-image + dark variants
    don't appear as separate entries in the catalogue UI.
    """
    results, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(
                key="store_name",
                match=models.MatchValue(value=store_name)
            )]
        ),
        limit=1000,   # fetch a large batch then deduplicate
        with_payload=True,
        with_vectors=False,
    )

    # Deduplicate by product_id — keep the first point seen per product
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
    """Delete a single product point from the catalogue by its Qdrant point ID."""
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

    async with httpx.AsyncClient() as http:
        vis_response = await http.post(
            f"{VISUAL_URL}/vectorize",
            files={"file": (file.filename, image_bytes, file.content_type)},
            data={"skip_rembg": "false", "yolo_label": "", "darken": "false"},
            timeout=60.0,
        )
        vis_response.raise_for_status()
        vis_data = vis_response.json()

    vector              = vis_data.get("vector")
    detected_category   = vis_data.get("category")
    category_confidence = vis_data.get("category_confidence", 0.0)
    processed_image     = vis_data.get("debug_image")

    query_filter    = None
    effective_label = search_label or detected_category
    if effective_label:
        query_filter = models.Filter(
            must=[models.FieldCondition(
                key="category_tag",
                match=models.MatchValue(value=effective_label)
            )]
        )

    # Fetch more candidates than needed so deduplication still returns 25
    raw_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        query_filter=query_filter,
        limit=100,
    )

    # ── Deduplicate by product_id — keep highest-scoring point per product ──
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
        "matches":             matches,
        "debug_image":         processed_image,
        "detected_category":   detected_category,
        "category_confidence": category_confidence,
    }


# ── Add Item (single, multipart form — legacy) ────────────────────────────────
@app.post("/add")
async def add_item(
    name:  str        = Form(...),
    store: str        = Form(...),
    mall:  str        = Form(...),
    file:  UploadFile = File(...),
):
    async with httpx.AsyncClient() as http_client:
        vis_response = await http_client.post(
            f"{VISUAL_URL}/vectorize",
            files={"file": (file.filename, await file.read(), file.content_type)},
            timeout=30.0,
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
                "image_url":    f"/static/{file.filename}",
                "category_tag": detected_category,
            }
        )]
    )
    return {"status": "saved", "item": name}


# ── Add Bulk Single (legacy JSON body) ────────────────────────────────────────
class BulkItem(BaseModel):
    name:       str
    store:      str
    mall:       str
    image_url:  str        = ""
    image_urls: list[str]  = []   # ← NEW: all image angles from Shopify
    price:      str        = ""
    category:   str        = ""
    product_id: str        = ""   # ← NEW: stable per-product identifier


@app.post("/add-bulk")
async def add_bulk_item(item: BulkItem):
    """Legacy single-item endpoint. Kept for backwards compatibility."""
    urls = item.image_urls if item.image_urls else ([item.image_url] if item.image_url else [])
    if not urls:
        raise HTTPException(status_code=400, detail="No image URL provided")

    product_id = item.product_id or str(uuid.uuid5(
        uuid.NAMESPACE_URL, f"{item.name}::{item.store}"
    ))

    inserted = 0
    async with httpx.AsyncClient() as http:
        for img_url in urls:
            img_bytes = (await http.get(img_url, timeout=15.0, follow_redirects=True)).content
            content_type = "image/jpeg"

            for darken in [False, True]:
                vis_resp = await http.post(
                    f"{VISUAL_URL}/vectorize",
                    files={"file": ("product.jpg", img_bytes, content_type)},
                    data={"darken": "true" if darken else "false"},
                    timeout=60.0,
                )
                vis_data          = vis_resp.json()
                vector            = vis_data.get("vector")
                detected_category = vis_data.get("category")
                final_category    = detected_category or item.category or "unknown"
                suffix            = "_dark" if darken else ""
                point_id          = str(uuid.uuid5(uuid.NAMESPACE_URL, img_url + suffix))

                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[PointStruct(
                        id      = point_id,
                        vector  = vector,
                        payload = {
                            "name":         item.name,
                            "store_name":   item.store,
                            "mall_name":    item.mall,
                            "image_url":    img_url,
                            "category_tag": final_category,
                            "price":        item.price,
                            "product_id":   product_id,
                            "is_dark":      darken,
                        }
                    )]
                )
                inserted += 1

    return {"status": "indexed", "item": item.name, "points_inserted": inserted}


# ══════════════════════════════════════════════════════════════════════════════
# /add-bulk-batch  — index many products in parallel
#
# Each item can now have image_urls (list).
# For every image URL we create 2 Qdrant points:
#   • normal vector  — id = uuid5(url)
#   • dark vector    — id = uuid5(url + "_dark")   brightness=0.3
# Both points share the same product_id payload so search can deduplicate.
# ══════════════════════════════════════════════════════════════════════════════

class BulkBatchRequest(BaseModel):
    items: list[dict]


@app.post("/add-bulk-batch")
async def add_bulk_batch(batch: BulkBatchRequest):
    semaphore = asyncio.Semaphore(10)

    async def index_one(raw: dict):
        name       = raw.get("name", "Product")
        store      = raw.get("store", "")
        mall       = raw.get("mall", "")
        price      = raw.get("price", "")
        category   = raw.get("category", "")

        # Accept both image_urls (list) and image_url (single string)
        image_urls = raw.get("image_urls") or []
        if not image_urls and raw.get("image_url"):
            image_urls = [raw["image_url"]]
        if not image_urls:
            return {"status": "failed", "item": name, "error": "no image URL"}

        # Stable product_id — same regardless of how many images the product has
        product_id = raw.get("product_id") or str(uuid.uuid5(
            uuid.NAMESPACE_URL, f"{name}::{store}"
        ))

        async with semaphore:
            try:
                async with httpx.AsyncClient() as http:
                    for img_url in image_urls:
                        # Fetch image bytes once, reuse for both normal + dark
                        img_resp = await http.get(img_url, timeout=15.0, follow_redirects=True)
                        img_resp.raise_for_status()
                        img_bytes    = img_resp.content
                        content_type = img_resp.headers.get("content-type", "image/jpeg")

                        for darken in [False, True]:
                            vis_resp = await http.post(
                                f"{VISUAL_URL}/vectorize",
                                files={"file": ("product.jpg", img_bytes, content_type)},
                                data={"darken": "true" if darken else "false"},
                                timeout=60.0,
                            )
                            vis_resp.raise_for_status()
                            vis_data = vis_resp.json()

                            vector            = vis_data.get("vector")
                            detected_category = vis_data.get("category")
                            final_category    = detected_category or category or "unknown"
                            suffix            = "_dark" if darken else ""
                            point_id          = str(uuid.uuid5(uuid.NAMESPACE_URL, img_url + suffix))

                            client.upsert(
                                collection_name=COLLECTION_NAME,
                                points=[PointStruct(
                                    id      = point_id,
                                    vector  = vector,
                                    payload = {
                                        "name":         name,
                                        "store_name":   store,
                                        "mall_name":    mall,
                                        "image_url":    img_url,
                                        "category_tag": final_category,
                                        "price":        price,
                                        "product_id":   product_id,
                                        "is_dark":      darken,
                                    }
                                )]
                            )

                return {"status": "ok", "item": name}

            except Exception as e:
                print(f"[BATCH] Failed: {name} — {e}")
                return {"status": "failed", "item": name, "error": str(e)}

    results = await asyncio.gather(*[index_one(raw) for raw in batch.items])
    success = sum(1 for r in results if r["status"] == "ok")
    failed  = [r for r in results if r["status"] == "failed"]
    return {"success": success, "total": len(batch.items), "failed": failed}


# ══════════════════════════════════════════════════════════════════════════════
# /scrape  — smart multi-strategy scraper
#
# Strategy order:
#   1. Shopify products.json API  — returns ALL image angles per product
#   2. JSON-LD structured data
#   3. Open Graph meta tags
#   4. Generic HTML product cards
# ══════════════════════════════════════════════════════════════════════════════

class ScrapeRequest(BaseModel):
    url:          str
    max_products: int = 0


@app.post("/scrape")
async def scrape_store(req: ScrapeRequest):
    req_headers = {
        "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept":          "application/json, text/html, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    parsed   = urlparse(req.url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"

    async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as http:

        # ── Strategy 1: Shopify ───────────────────────────────────────────────
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
                    "name":       (og_title["content"] if og_title else "Product").strip(),
                    "image_url":  og_image["content"].strip(),
                    "image_urls": [og_image["content"].strip()],
                    "price":      (og_price["content"] if og_price else ""),
                })

        # Strategy 4: HTML cards
        if not products:
            products = _scrape_product_cards(soup, req.url)

    # Deduplicate
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


# ── Shopify API helper ─────────────────────────────────────────────────────────
async def _try_shopify_api(
    http: httpx.AsyncClient,
    original_url: str,
    base_url: str,
    max_products: int,
    headers: dict,
) -> list:
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
                print(f"[SCRAPE] Shopify API page {page}: {api_url}")

                resp = await http.get(api_url, headers=headers, timeout=15.0)
                if resp.status_code != 200:
                    break

                raw_products = resp.json().get("products", [])
                if not raw_products:
                    break

                for p in raw_products:
                    name = p.get("title", "").strip()
                    if not name:
                        continue

                    # ── Collect ALL image angles ──────────────────────────────
                    images     = p.get("images", [])
                    image_urls = [img.get("src", "") for img in images if img.get("src")]
                    image_url  = image_urls[0] if image_urls else ""

                    if not image_url:
                        continue

                    price    = ""
                    variants = p.get("variants", [])
                    if variants:
                        price_val = variants[0].get("price", "")
                        if price_val:
                            price = f"USD {price_val}"

                    all_products.append({
                        "name":       name,
                        "image_url":  image_url,    # first image (used for preview card)
                        "image_urls": image_urls,   # ALL images (used for indexing)
                        "price":      price,
                    })

                print(f"[SCRAPE] Page {page}: {len(raw_products)} products (total: {len(all_products)})")

                if max_products and len(all_products) >= max_products:
                    all_products = all_products[:max_products]
                    break

                if len(raw_products) < 250:
                    break

                page += 1

            if all_products:
                print(f"[SCRAPE] Shopify complete: {len(all_products)} products")
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
    price  = ""
    offers = obj.get("offers", {})
    if isinstance(offers, list): offers = offers[0] if offers else {}
    if isinstance(offers, dict):
        price_val = str(offers.get("price", ""))
        currency  = offers.get("priceCurrency", "")
        if price_val and currency:
            price = f"{currency} {price_val}"
    if name and image:
        image_str = str(image).strip()
        products.append({
            "name":       name,
            "image_url":  image_str,
            "image_urls": [image_str],
            "price":      price,
        })


def _scrape_product_cards(soup: BeautifulSoup, base_url: str) -> list:
    products = []
    cards    = soup.select("[class*='product'], [class*='item'], article")
    for card in cards[:50]:
        img = card.find("img")
        if not img:
            continue
        src = img.get("src") or img.get("data-src", "")
        if not src or src.startswith("data:"):
            continue
        if not src.startswith("http"):
            src = base_url.rstrip("/") + "/" + src.lstrip("/")
        title_el = card.find(["h2", "h3", "h4", "a", "[class*='title']", "[class*='name']"])
        name     = title_el.get_text(strip=True) if title_el else "Product"
        if len(name) > 120:
            continue
        products.append({
            "name":       name,
            "image_url":  src,
            "image_urls": [src],
            "price":      "",
        })
    return products