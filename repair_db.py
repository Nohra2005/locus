from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

client = QdrantClient("localhost", port=6333)
COLLECTION = "locus_items"

if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)
    print("🗑️  Deleted old collection")

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
)
print("✅ Fresh collection created. Ready for bulk_upload.py")