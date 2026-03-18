import asyncio
import io
import json as _json
import os
import uuid
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

VISUAL_URL      = os.getenv("VISUAL_HOST",    "http://visual_engine:8001")
QDRANT_URL      = os.getenv("QDRANT_URL")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")
QDRANT_HOST     = os.getenv("QDRANT_HOST",    "qdrant")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "locus_items"

if QDRANT_URL:
    print(f"[QDRANT] Connecting to cloud: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    print(f"[QDRANT] Connecting to local: {QDRANT_HOST}:{QDRANT_PORT}")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


@app.on_event("startup")
def startup_event():
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
    # Required for filtered search on Qdrant Cloud
    for field in ("category_tag", "store_name", "product_id"):
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema="keyword",
            )
        except Exception:
            pass  # index already exists


@app.get("/")
def read_root():
    return {"status": "online", "service": "Locus Gateway"}


# ── Crop helper ───────────────────────────────────────────────────────────────
def _crop_image_bytes(image_bytes: bytes, x1: float, y1: float, x2: float, y2: float) -> bytes:
    img  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W, H = img.size
    x1c = max(0, int(x1)); y1c = max(0, int(y1))
    x2c = min(W, int(x2)); y2c = min(H, int(y2))
    crop = img.crop((x1c, y1c, x2c, y2c))
    buf  = io.BytesIO()
    crop.save(buf, format="JPEG")
    return buf.getvalue()


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
        "matches":             matches,
        "debug_image":         processed_image,
        "detected_category":   detected_category,
        "category_confidence": category_confidence,
    }


# ══════════════════════════════════════════════════════════════════════════════
# /add-bulk-batch
# ══════════════════════════════════════════════════════════════════════════════

class BulkBatchRequest(BaseModel):
    items: list[dict]


@app.post("/add-bulk-batch")
async def add_bulk_batch(batch: BulkBatchRequest):
    semaphore = asyncio.Semaphore(5)

    async def index_one(raw: dict):
        name     = raw.get("name", "Product")
        store    = raw.get("store", "")
        mall     = raw.get("mall", "")
        price    = raw.get("price", "")

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
                        return {"status": "skipped", "item": name, "reason": reason}

                    vector_normal  = idx_data["vector_normal"]
                    vector_dark    = idx_data["vector_dark"]
                    final_category = idx_data.get("category", "unknown")

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
                                    "is_dark":      False,
                                }
                            ),
                            PointStruct(
                                id      = str(uuid.uuid5(uuid.NAMESPACE_URL, img_url + "_dark")),
                                vector  = vector_dark,
                                payload = {
                                    "name":         name,
                                    "store_name":   store,
                                    "mall_name":    mall,
                                    "image_url":    img_url,
                                    "category_tag": final_category,
                                    "price":        price,
                                    "product_id":   product_id,
                                    "is_dark":      True,
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

        # HTML fallback
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

                # Guard against compressed or non-JSON responses
                try:
                    import gzip as _gzip
                    content = resp.content
                    # Try gzip decompression if needed
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