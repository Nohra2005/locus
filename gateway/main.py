import os
import uuid
import httpx  # Async replacement for requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles  # <--- NEW: For serving images
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

app = FastAPI()

# --- NEW: Serve the demo_images folder so the Frontend can see them ---
# We assume the 'demo_images' folder is one level up or mapped in Docker.
# If this fails inside Docker, ensure your docker-compose maps the volume correctly.
try:
    # Try mounting the local folder relative to this script
    if os.path.exists("../demo_images"):
        app.mount("/static", StaticFiles(directory="../demo_images"), name="static")
        print("✅ Static file serving enabled for ../demo_images")
    elif os.path.exists("./demo_images"):
        app.mount("/static", StaticFiles(directory="./demo_images"), name="static")
        print("✅ Static file serving enabled for ./demo_images")
    else:
        print("⚠️ Warning: 'demo_images' folder not found. Images won't load in UI.")
except Exception as e:
    print(f"⚠️ Could not mount static files: {e}")

# Configuration
VISUAL_URL = "http://visual_engine:8001/vectorize"
QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333
COLLECTION_NAME = "locus_items"

# Initialize Qdrant Client
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

@app.on_event("startup")
def startup_event():
    """
    On startup, ensure collection exists.
    """
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
        print(f"✅ Created collection: {COLLECTION_NAME}")

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
    """
    Vectorizes the image and saves it with FULL metadata.
    """
    # --- Step 1: Vectorize (Async) ---
    async with httpx.AsyncClient() as http_client:
        files = {"file": (file.filename, await file.read(), file.content_type)}
        try:
            # Increased timeout for safety
            vis_response = await http_client.post(VISUAL_URL, files=files, timeout=30.0)
            vis_response.raise_for_status()
            vector = vis_response.json().get("vector")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Visual Engine Error: {e}")

    # --- Step 2: Save to Qdrant ---
    point_id = str(uuid.uuid4())
    
    payload = {
        "name": name,
        "store_name": store,
        "floor_level": level,
        "mall_name": mall,
        "filename": file.filename
    }

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
        ]
    )
    
    print(f"💾 Indexed: {name} at {store} ({level})")
    return {
        "status": "saved", 
        "id": point_id, 
        "item": name, 
        "location": f"{store}, {level}"
    }

@app.post("/search")
async def search(file: UploadFile = File(...)):
    # 1. Vectorize Query (and get the debug image)
    async with httpx.AsyncClient() as http_client:
        files = {"file": (file.filename, await file.read(), file.content_type)}
        # Increase timeout because we are sending back image data
        vis_response = await http_client.post(VISUAL_URL, files=files, timeout=40.0)
        
        data = vis_response.json()
        query_vector = data.get("vector")
        processed_image = data.get("processed_image") # <--- Capture it

    # 2. Search Qdrant
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=25
    )
    
    # 3. Format matches
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
        "count": len(matches),
        "matches": matches,
        "debug_image": processed_image # <--- Send to Frontend
    }