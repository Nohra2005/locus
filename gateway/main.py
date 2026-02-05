import os
import uuid
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from qdrant_client.http import models # <--- Logic for filtering
from qdrant_client.http.models import Distance, VectorParams, PointStruct

app = FastAPI()

# Serve Images
try:
    if os.path.exists("../demo_images"):
        app.mount("/static", StaticFiles(directory="../demo_images"), name="static")
    elif os.path.exists("./demo_images"):
        app.mount("/static", StaticFiles(directory="./demo_images"), name="static")
except Exception:
    pass

# Config
VISUAL_URL = "http://visual_engine:8001/vectorize"
QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333
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

@app.post("/add")
async def add_item(
    name: str = Form(...),
    store: str = Form(...),
    level: str = Form(...),
    mall: str = Form(...),
    file: UploadFile = File(...)
):
    async with httpx.AsyncClient() as http_client:
        files = {"file": (file.filename, await file.read(), file.content_type)}
        vis_response = await http_client.post(VISUAL_URL, files=files, timeout=30.0)
        vis_response.raise_for_status()
        data = vis_response.json()
        vector = data.get("vector")
        detected_category = data.get("category")

    point_id = str(uuid.uuid4())
    payload = {
        "name": name, "store_name": store, "floor_level": level, 
        "mall_name": mall, "filename": file.filename, 
        "category_tag": detected_category
    }

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(id=point_id, vector=vector, payload=payload)]
    )
    return {"status": "saved", "item": name}

@app.post("/search")
async def search(file: UploadFile = File(...)):
    # 1. Vectorize Query
    async with httpx.AsyncClient() as http_client:
        files = {"file": (file.filename, await file.read(), file.content_type)}
        vis_response = await http_client.post(VISUAL_URL, files=files, timeout=40.0)
        
        data = vis_response.json()
        query_vector = data.get("vector")
        processed_image = data.get("processed_image") # Partner's Debug Image
        detected_category = data.get("category")      # Your Category

    # 2. Build Filter (YOUR LOGIC)
    query_filter = None
    if detected_category:
        print(f"🎯 Filter: {detected_category}")
        query_filter = models.Filter(
            should=[models.FieldCondition(key="name", match=models.MatchText(text=detected_category))]
        )

    # 3. Search Qdrant
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=query_filter, # <--- Apply Filter
        limit=25
    )
    
    matches = []
    for hit in search_result:
        matches.append({
            "name": hit.payload.get("name", "Unknown"),
            "store": hit.payload.get("store_name", "Unknown"),
            "level": hit.payload.get("floor_level", "Unknown"),
            "mall": hit.payload.get("mall_name", "Unknown"),
            "score": round(hit.score, 3), 
            "image_filename": hit.payload.get("filename")
        })
        
    return {
        "matches": matches,
        "debug_image": processed_image,      # Send to UI
        "detected_category": detected_category # Send to UI
    }