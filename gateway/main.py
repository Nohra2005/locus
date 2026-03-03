import os
import uuid
import io
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from PIL import Image

app = FastAPI()

# ── Serve DeepFashion dataset images at /static ──────────────────────────────
# Add this volume to docker-compose.yml under the gateway service:
#   volumes:
#     - "C:/Users/User/.cache/kagglehub/datasets/hserdaraltan/deepfashion-inshop-clothes-retrieval/versions/1:/app/dataset"
# bulk_upload stores URLs as /static/img_highres/... so we mount at /static
DATASET_DIR = os.environ.get("DATASET_DIR", "/app/dataset")
try:
    if os.path.exists(DATASET_DIR):
        app.mount("/static", StaticFiles(directory=DATASET_DIR), name="static")
        print(f"Serving dataset images from {DATASET_DIR}")
    else:
        print(f"WARNING: DATASET_DIR not found: {DATASET_DIR}")
except Exception as e:
    print(f"WARNING: Could not mount static files: {e}")

# ── Config ────────────────────────────────────────────────────────────────────
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

@app.get("/")
def read_root():
    return {"status": "online", "service": "Locus Gateway"}

# ── Feedback ──────────────────────────────────────────────────────────────────
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

# ── Health Check ──────────────────────────────────────────────────────────────
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

# ── Step 1: Detect ────────────────────────────────────────────────────────────
@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    async with httpx.AsyncClient() as http_client:
        files    = {"file": (file.filename, await file.read(), file.content_type)}
        response = await http_client.post(f"{VISUAL_URL}/detect", files=files, timeout=60.0)
        response.raise_for_status()
        return response.json()

# ── Step 2: Search ────────────────────────────────────────────────────────────
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
        image_bytes  = buf.getvalue()
        filename     = "cropped_selection.png"
        content_type = "image/png"
    else:
        filename     = file.filename
        content_type = file.content_type

    async with httpx.AsyncClient() as http_client:
        vis_response = await http_client.post(
            f"{VISUAL_URL}/vectorize",
            files={"file": (filename, image_bytes, content_type)},
            data={
                "skip_rembg": "true" if is_cropped else "false",
                "yolo_label": search_label if search_label else ""
            },
            timeout=40.0
        )
        data                = vis_response.json()
        query_vector        = data.get("vector")
        processed_image     = data.get("debug_image")
        detected_category   = data.get("category")
        category_confidence = data.get("category_confidence", 1.0)

    if not query_vector:
        raise HTTPException(status_code=400, detail="Could not vectorize image")

    if search_label:
        detected_category = search_label

    # No category filter — CLIP labels ("shirt") never match index tags
    # ("blouses shirts"). Pure vector similarity works better.
    # Fetch 50 so the frontend can deduplicate multi-view/dim duplicates.
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=None,
        limit=50
    )

    matches = []
    for hit in search_result:
        matches.append({
            "name":           hit.payload.get("name", "Unknown"),
            "store":          hit.payload.get("store_name", "Unknown"),
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

# ── Add Item ──────────────────────────────────────────────────────────────────
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