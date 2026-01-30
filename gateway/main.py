import requests
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid

app = FastAPI()

# Configuration
VISUAL_URL = "http://visual_engine:8001/vectorize"
# We connect to the Qdrant service we defined in docker-compose
QDRANT_HOST = "qdrant" 
QDRANT_PORT = 6333
COLLECTION_NAME = "locus_items"

# Initialize Qdrant Client
# This client handles all the communication with the vector database
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

@app.on_event("startup")
def startup_event():
    """
    On startup, we ensure the collection exists.
    If not, we create it with a vector size of 512 (Standard for CLIP).
    """
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
        print(f"Created collection: {COLLECTION_NAME}")

@app.get("/")
def read_root():
    return {"status": "online", "service": "Locus Gateway"}

@app.post("/add")
async def add_item(name: str, file: UploadFile = File(...)):
    """
    1. Vectorize the uploaded image.
    2. Save the (Vector + Name) into Qdrant.
    """
    # --- Step 1: Vectorize ---
    files = {"file": (file.filename, file.file, file.content_type)}
    try:
        vis_response = requests.post(VISUAL_URL, files=files)
        vector = vis_response.json().get("vector")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visual Engine Error: {e}")

    # --- Step 2: Save to Qdrant ---
    # We generate a random ID for the database entry
    point_id = str(uuid.uuid4())
    
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload={"name": name, "filename": file.filename}
            )
        ]
    )
    
    return {"status": "saved", "id": point_id, "name": name}

@app.post("/search")
async def search(file: UploadFile = File(...)):
    """
    1. Vectorize the uploaded image.
    2. Search Qdrant for the nearest vectors.
    """
    # --- Step 1: Vectorize ---
    files = {"file": (file.filename, file.file, file.content_type)}
    try:
        vis_response = requests.post(VISUAL_URL, files=files)
        query_vector = vis_response.json().get("vector")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visual Engine Error: {e}")

    # --- Step 2: Search Qdrant ---
    # We ask for the top 3 closest matches
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=3
    )
    
    # Format the results
    matches = []
    for hit in search_result:
        matches.append({
            "name": hit.payload["name"],
            "score": hit.score,
            "filename": hit.payload["filename"]
        })
        
    return {
        "query_image": file.filename,
        "matches": matches
    }