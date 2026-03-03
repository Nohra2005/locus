"""
bulk_upload_deepfashion.py

Indexes DeepFashion In-Shop Retrieval dataset into Qdrant.

Image serving strategy:
  The gateway mounts the demo_images folder at /static.
  We mount the DeepFashion dataset INTO that same folder via Docker volume
  (see docker-compose.yml). So images are served at /static/img/...
  No gateway changes needed.

Vector strategy:
  Per image we store TWO vectors:
    1. Bright — raw pixels, matches well-lit query photos
    2. Dim    — 0.65 brightness, matches poorly-lit query photos
  Both point to the same original /static image_url (always the bright photo).

Payload field names match what gateway/main.py reads exactly:
  - filename     → gateway reads hit.payload.get("filename")
  - store_name   → gateway reads hit.payload.get("store_name")
  - category_tag → gateway filters on key="category_tag"
"""

import os
import uuid
from collections import defaultdict

import torch
from PIL import Image, ImageEnhance
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import kagglehub

# Dynamically get the dataset path — no hardcoding needed
DATASET_PATH = kagglehub.dataset_download("hserdaraltan/deepfashion-inshop-clothes-retrieval")
IMAGE_DIR    = os.path.join(DATASET_PATH, "img_highres")
COLLECTION_NAME  = "locus_items"   # must match gateway COLLECTION_NAME
MAX_PER_CATEGORY = 150             # item cap per category for balance
DIM_FACTOR       = 0.65            # brightness for dim vectors
BATCH_SIZE       = 64

# ── Category → store_name mapping ────────────────────────────────────
# store_name matches what gateway returns to frontend as "store"
CATEGORY_TO_STORE = {
    "Blouses_Shirts":      "Zara",
    "Dresses":             "Zara",
    "Jackets_Vests":       "Zara",
    "Skirts":              "Zara",
    "Sweaters":            "Zara",
    "Cardigans":           "Zara",
    "Suiting":             "Zara",
    "Rompers_Jumpsuits":   "Zara",
    "Sweatshirts_Hoodies": "Bershka",
    "Denim":               "Bershka",
    "Pants_Capris":        "Bershka",
    "Shorts":              "Bershka",
    "Tees_Tanks":          "Bershka",
    "Graphic_Tees":        "Bershka",
    "Leggings":            "Bershka",
    "Shirts_Polos":        "Bershka",
    "Activewear":          "Mike Sport",
    "Accessories":         "Virgin",
}

# Category tag values must match what visual_engine/vectorizer.py outputs
# so the gateway filter (key="category_tag") works correctly
def get_store(category: str) -> str:
    for key, store in CATEGORY_TO_STORE.items():
        if key.lower() in category.lower():
            return store
    return "Zara"

def get_tag(category: str) -> str:
    # Use the folder name directly — already specific.
    # e.g. "Sunglasses", "Hats", "Bags", "Dresses", "Jackets_Vests"
    # Lowercase + replace underscores to match CLIP label style.
    return category.lower().replace("_", " ")


# ── Qdrant ────────────────────────────────────────────────────────────
qdrant = QdrantClient("localhost", port=6333)

if qdrant.collection_exists(COLLECTION_NAME):
    count = qdrant.get_collection(COLLECTION_NAME).points_count
    print(f"Collection '{COLLECTION_NAME}' already has {count} points.")
    ans = input("Delete and recreate? [y/N] ").strip().lower()
    if ans == "y":
        qdrant.delete_collection(COLLECTION_NAME)
        print("Deleted old collection.")
    else:
        print("Appending to existing collection.")

if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )
    print(f"Created '{COLLECTION_NAME}'.")


# ── CLIP — same model as visual_engine/vectorizer.py ─────────────────
print("Loading CLIP...")
device    = "cuda" if torch.cuda.is_available() else "cpu"
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
model.eval()
print(f"CLIP ready on {device}.\n")


def embed(img: Image.Image) -> list:
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        f = model.get_image_features(**inputs)
    # newer transformers versions return an object, not a tensor directly
    if not isinstance(f, torch.Tensor):
        f = f.image_embeds if hasattr(f, "image_embeds") else f.pooler_output
    f = f / f.norm(p=2, dim=-1, keepdim=True)
    return f[0].cpu().tolist()

def darken(img: Image.Image) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(DIM_FACTOR)


# ── Collect items grouped by item_id ─────────────────────────────────
print("Scanning dataset...")
items_by_category = defaultdict(list)
seen_items        = defaultdict(set)

for root, _, files in os.walk(IMAGE_DIR):
    parts = root.replace("\\", "/").split("/")
    try:
        img_idx  = parts.index("img_highres")
        gender   = parts[img_idx + 1]
        category = parts[img_idx + 2]
        item_id  = parts[img_idx + 3]
    except (ValueError, IndexError):
        continue

    if len(seen_items[category]) >= MAX_PER_CATEGORY:
        continue

    # Exclude segment masks
    images = [
        os.path.join(root, f)
        for f in files
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and "segment" not in f.lower()
    ]
    if not images:
        continue

    if item_id not in seen_items[category]:
        seen_items[category].add(item_id)
        items_by_category[category].append({
            "item_id":  item_id,
            "gender":   gender,
            "images":   images,
        })

total_items  = sum(len(v) for v in items_by_category.values())
total_images = sum(len(e["images"]) for entries in items_by_category.values() for e in entries)
print(f"Items   : {total_items}")
print(f"Images  : {total_images}")
print(f"Vectors : {total_images * 2} (bright + dim)\n")
print("Category breakdown:")
for cat, entries in sorted(items_by_category.items()):
    n = sum(len(e["images"]) for e in entries)
    print(f"  {cat:<30} {len(entries)} items · {n} images · {n*2} vectors")
print()


# ── Upload ────────────────────────────────────────────────────────────
points  = []
skipped = 0

all_entries = [
    (cat, entry)
    for cat, entries in items_by_category.items()
    for entry in entries
]

for category, entry in tqdm(all_entries, desc="Uploading"):
    item_id   = entry["item_id"]
    gender    = entry["gender"]
    store     = get_store(category)
    tag       = get_tag(category)
    item_name = f"{gender} {category.replace('_', ' ')}"

    for img_path in entry["images"]:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"\nSkip {img_path}: {e}")
            skipped += 1
            continue

        # URL served by gateway /static mount.
        # Docker volume maps DATASET_PATH → /app/demo_images in container,
        # which the gateway mounts at /static.
        # So:  DATASET_PATH/img/WOMEN/Dresses/id/file.jpg
        #   →  /static/img/WOMEN/Dresses/id/file.jpg
        relative  = img_path.replace(DATASET_PATH, "").replace("\\", "/")
        if not relative.startswith("/"):
            relative = "/" + relative
        static_url = f"/static{relative}"   # gateway serves this directly

        # Payload field names must match what gateway/main.py reads:
        #   hit.payload.get("filename")   → shown as image in frontend
        #   hit.payload.get("store_name") → shown as store in frontend
        #   key="category_tag"            → used for Qdrant filter
        base_payload = {
            "filename":     static_url,   # gateway reads "filename"
            "store_name":   store,        # gateway reads "store_name"
            "category_tag": tag,          # gateway filters on "category_tag"
            "name":         item_name,    # gateway reads "name"
            "item_id":      item_id,      # frontend feedback uses item_id
            "gender":       gender,
            "category":     category,
        }

        # Vector 1 — bright (matches well-lit queries)
        try:
            points.append(PointStruct(
                id      = str(uuid.uuid4()),
                vector  = embed(img),
                payload = {**base_payload, "lighting": "bright"},
            ))
        except Exception as e:
            print(f"\nBright embed failed {img_path}: {e}")
            skipped += 1

        # Vector 2 — dim (matches poorly-lit queries)
        # image_url still points to bright photo — only the vector differs
        try:
            points.append(PointStruct(
                id      = str(uuid.uuid4()),
                vector  = embed(darken(img)),
                payload = {**base_payload, "lighting": "dim"},
            ))
        except Exception as e:
            print(f"\nDim embed failed {img_path}: {e}")

        if len(points) >= BATCH_SIZE:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            points = []

if points:
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

final = qdrant.get_collection(COLLECTION_NAME).points_count
print(f"\nDone. {final} vectors in '{COLLECTION_NAME}'. Skipped {skipped} images.")