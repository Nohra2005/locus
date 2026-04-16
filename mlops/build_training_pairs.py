"""
build_training_pairs.py — Build augmentation-based training pairs for LoRA fine-tuning.

Strategy:
  For each product image in locus_items, we create N augmented versions that
  simulate how the same item looks in a consumer/internet photo (variable
  lighting, partial crops, blur, perspective shift). Each (augmented, original)
  pair teaches CLIP: "these are the same item, pull them together."

  Feedback signal from locus_feedback is used to weight sampling:
    - Products with positive feedback (rating >= 4) get 3 augmented pairs
    - Products with negative feedback (rating <= 2) get 1 pair (still informative)
    - Unrated products get 2 pairs

  Output: pairs_cache/ directory with anchor_*.jpg and positive_*.jpg files,
          plus pairs_manifest.json listing all pairs and their weights.

Usage:
  python build_training_pairs.py
  QDRANT_URL=... QDRANT_API_KEY=... python build_training_pairs.py
"""

import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

from PIL import Image, ImageFilter
import random
import io

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_URL        = os.getenv("QDRANT_URL")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY")
QDRANT_HOST       = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME   = "locus_items"
FEEDBACK_COL      = "locus_feedback"
CACHE_DIR         = Path(os.getenv("PAIRS_CACHE_DIR", "pairs_cache"))
MAX_PRODUCTS      = int(os.getenv("MAX_PRODUCTS", 400))   # cap to keep training time <2h on CPU
TARGET_SIZE       = (224, 224)
# ─────────────────────────────────────────────────────────────────────────────


def _get_qdrant_client():
    from qdrant_client import QdrantClient
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def _fetch_image(url: str, timeout: int = 15) -> Optional[Image.Image]:
    """Download an image from URL. Returns None on failure."""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (Locus-Trainer/1.0)"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return img
    except Exception as e:
        print(f"  [SKIP] Could not fetch {url[:60]}: {e}")
        return None


def _augment(img: Image.Image, seed: int) -> Image.Image:
    """
    Apply a deterministic set of augmentations to simulate how the same item
    looks in a consumer/internet photo:
      - Random crop (simulates partial view / zoom)
      - Color jitter (lighting variation)
      - Optional blur (camera focus issues)
      - Optional horizontal flip
      - Optional slight rotation

    We use PIL only (no torchvision dependency here) to keep requirements lean.
    """
    random.seed(seed)
    w, h = img.size

    # 1. Random crop — keep between 50% and 95% of each dimension
    crop_ratio_w = random.uniform(0.50, 0.95)
    crop_ratio_h = random.uniform(0.50, 0.95)
    cw = int(w * crop_ratio_w)
    ch = int(h * crop_ratio_h)
    left   = random.randint(0, w - cw)
    top    = random.randint(0, h - ch)
    img    = img.crop((left, top, left + cw, top + ch))

    # 2. Resize back to target
    img = img.resize(TARGET_SIZE, Image.BILINEAR)

    # 3. Color jitter via point transform
    # brightness
    brightness = random.uniform(0.6, 1.4)
    img = img.point(lambda p: min(255, int(p * brightness)))

    # contrast (blend with gray)
    contrast = random.uniform(0.7, 1.3)
    gray_val = 128
    img = img.point(lambda p: int(gray_val + (p - gray_val) * contrast))
    img = img.point(lambda p: max(0, min(255, p)))

    # 4. Optional Gaussian blur (simulate phone camera blur)
    if random.random() < 0.4:
        radius = random.uniform(0.5, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # 5. Optional horizontal flip
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    return img


def _build_feedback_weights(client) -> dict[str, int]:
    """
    Return {product_id: augmentation_count} based on feedback signals.
    Positive feedback → 3 augmented pairs (model found these images useful).
    Negative feedback → 1 augmented pair (still informative as hard negatives).
    Default (no feedback) → 2 augmented pairs.
    """
    weights: dict[str, int] = {}
    try:
        offset = None
        while True:
            batch, next_offset = client.scroll(
                collection_name=FEEDBACK_COL,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not batch:
                break
            for pt in batch:
                pid    = pt.payload.get("result_product_id", "")
                signal = pt.payload.get("training_signal", "neutral")
                if not pid:
                    continue
                current = weights.get(pid, 2)  # default 2
                if signal == "positive":
                    weights[pid] = max(current, 3)
                elif signal == "negative":
                    weights[pid] = min(current, 1)
            if next_offset is None:
                break
            offset = next_offset
        print(f"[PAIRS] Feedback loaded: {len(weights)} products have ratings")
    except Exception as e:
        print(f"[PAIRS] Could not load feedback (using default weights): {e}")
    return weights


def build_pairs(max_products: int = MAX_PRODUCTS) -> Path:
    """
    Download up to max_products catalog images and build augmented training pairs.
    Returns path to pairs_manifest.json.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    manifest_path = CACHE_DIR / "pairs_manifest.json"

    client = _get_qdrant_client()

    # Load feedback weights
    feedback_weights = _build_feedback_weights(client)

    # Scroll locus_items for product image URLs
    print(f"[PAIRS] Fetching up to {max_products} products from Qdrant...")
    products = []
    offset   = None
    while len(products) < max_products:
        batch, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=min(250, max_products - len(products)),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not batch:
            break
        for pt in batch:
            url = pt.payload.get("image_url", "")
            pid = pt.payload.get("product_id", str(pt.id))
            if url:
                products.append({"product_id": pid, "image_url": url})
        if next_offset is None:
            break
        offset = next_offset

    print(f"[PAIRS] Found {len(products)} products with image URLs")

    manifest = []
    skipped  = 0

    for i, product in enumerate(products):
        pid = product["product_id"]
        url = product["image_url"]
        n_augments = feedback_weights.get(pid, 2)

        # Download original
        orig_path = CACHE_DIR / f"orig_{i:04d}.jpg"
        if not orig_path.exists():
            img = _fetch_image(url)
            if img is None:
                skipped += 1
                continue
            img = img.resize(TARGET_SIZE, Image.BILINEAR)
            img.save(orig_path, "JPEG", quality=90)
            time.sleep(0.05)  # polite crawl delay
        else:
            img = Image.open(orig_path).convert("RGB")

        # Create N augmented versions
        for aug_idx in range(n_augments):
            aug_path = CACHE_DIR / f"aug_{i:04d}_{aug_idx}.jpg"
            if not aug_path.exists():
                # Different seed per augmentation → different transforms
                seed = i * 100 + aug_idx
                aug_img = _augment(img, seed=seed)
                aug_img.save(aug_path, "JPEG", quality=85)

            manifest.append({
                "anchor_path":   str(aug_path),    # augmented = "consumer photo"
                "positive_path": str(orig_path),   # original  = "catalog photo"
                "product_id":    pid,
                "aug_index":     aug_idx,
                "weight":        1.0 if n_augments == 3 else (0.6 if n_augments == 1 else 0.8),
            })

        if (i + 1) % 50 == 0:
            print(f"[PAIRS] Processed {i + 1}/{len(products)} products "
                  f"({len(manifest)} pairs so far)...")

    # Shuffle pairs so training sees diverse products each epoch
    random.shuffle(manifest)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[PAIRS] Done.")
    print(f"  Products processed : {len(products) - skipped}")
    print(f"  Products skipped   : {skipped}")
    print(f"  Training pairs     : {len(manifest)}")
    print(f"  Manifest saved to  : {manifest_path}")

    return manifest_path


if __name__ == "__main__":
    build_pairs()
