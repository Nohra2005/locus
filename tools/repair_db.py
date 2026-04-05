"""
repair_db.py — Wipes and recreates the locus_items Qdrant collection.

⚠️  WARNING: This deletes ALL indexed products. You will need to re-run bulk_upload.py
    to rebuild the catalog from scratch. Only use this if the collection is corrupted
    or you are intentionally resetting the index.

Usage:
    python3 repair_db.py           # prompts for confirmation
    python3 repair_db.py --confirm # skip prompt (for scripted use only)
"""
import argparse
import sys

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

parser = argparse.ArgumentParser()
parser.add_argument("--confirm", action="store_true", help="Skip confirmation prompt")
args = parser.parse_args()

COLLECTION = "locus_items"

if not args.confirm:
    print("⚠️  WARNING: This will permanently delete ALL indexed products in Qdrant.")
    print(f"   Collection: {COLLECTION}")
    answer = input("   Type 'yes' to continue, anything else to abort: ").strip().lower()
    if answer != "yes":
        print("Aborted.")
        sys.exit(0)

client = QdrantClient("localhost", port=6333)

if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)
    print("🗑️  Deleted old collection")

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
)
print("✅ Fresh collection created. Ready for bulk_upload.py")
