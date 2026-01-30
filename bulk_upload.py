import requests
import os
import json
from qdrant_client import QdrantClient
from qdrant_client.http import models

# --- Config ---
API_URL = "http://localhost:8000/add"
IMAGE_FOLDER = "./demo_images"
MALL_CONFIG = "mall_config.json"
MALL_NAME = "ABC Achrafieh"

# Connect to Qdrant directly to check for duplicates
# (This assumes Qdrant is running on localhost:6333)
qdrant = QdrantClient("localhost", port=6333)

def get_store_info(filename, directory):
    # Heuristic: Split "zara_dress.jpg" -> "Zara"
    store_key = filename.split('_')[0].capitalize()
    
    # Handle "mike_sport" -> "Mike"
    if store_key == "Mike": store_key = "Mike" # Mapping logic if needed
    
    # Lookup in config
    return store_key, directory.get(store_key, {"level": "L1"})

def is_already_indexed(filename):
    """
    Asks Qdrant: 'Do you have an item with this filename?'
    """
    try:
        result, _ = qdrant.scroll(
            collection_name="locus_items",
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="filename",
                        match=models.MatchValue(value=filename)
                    )
                ]
            ),
            limit=1
        )
        return len(result) > 0
    except Exception:
        # If collection doesn't exist yet, it's not indexed
        return False

def run_upload():
    # Load Mall Config
    if not os.path.exists(MALL_CONFIG):
        print("❌ Error: mall_config.json not found!")
        return

    with open(MALL_CONFIG) as f:
        directory = json.load(f).get(MALL_NAME, {})

    print(f"🚀 Starting Smart Upload for {MALL_NAME}...")

    # Iterate images
    files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    new_count = 0
    skip_count = 0

    for filename in files:
        # 1. CHECK: Is it already there?
        if is_already_indexed(filename):
            print(f"⏭️  Skipping {filename} (Already exists)")
            skip_count += 1
            continue

        # 2. PREPARE: metadata
        store_key, store_info = get_store_info(filename, directory)
        metadata = {
            "name": filename.replace("_", " ").split('.')[0],
            "store": store_key,
            "level": store_info['level'],
            "mall": MALL_NAME
        }
        
        # 3. UPLOAD: Send to AI
        print(f"📤 Uploading {filename}...")
        file_path = os.path.join(IMAGE_FOLDER, filename)
        with open(file_path, "rb") as img:
            try:
                # Use 'data' for form fields, 'files' for the image
                r = requests.post(API_URL, data=metadata, files={"file": img}, timeout=30)
                if r.status_code == 200:
                    print(f"   ✅ Success")
                    new_count += 1
                else:
                    print(f"   ❌ Failed: {r.text}")
            except Exception as e:
                print(f"   ❌ Error: {e}")

    print(f"\n🏁 Complete! Added {new_count} new items. Skipped {skip_count} duplicates.")

if __name__ == "__main__":
    run_upload()