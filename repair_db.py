from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# Connect to local Qdrant
client = QdrantClient("localhost", port=6333)
COLLECTION_NAME = "locus_items"

print(f"🔧 Attempting to repair: {COLLECTION_NAME}...")

if not client.collection_exists(collection_name=COLLECTION_NAME):
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
        print(f"✅ Success: Created collection '{COLLECTION_NAME}'")
    except Exception as e:
        print(f"❌ Error creating collection: {e}")
else:
    print(f"⚠️ Collection '{COLLECTION_NAME}' already exists. No action needed.")

print("🚀 Database is ready. You can run bulk_upload.py now.")