"""
Golden Dataset Annotation Tool — Locus Visual Search
=====================================================
Run this script to interactively build your golden dataset for MLFlow experiments.

What it does:
  1. You give it a Qdrant point UUID from your locus_items collection
  2. It fetches that product's info and finds the 20 most similar *unique* products
     (deduplicating normal vs dark variants — each real product shows up once)
  3. You pick exactly 5 of those as the ground-truth matches
  4. It saves your annotation to golden_dataset.json using product_id as the key

Usage:
  python annotate_golden.py

  # To connect to Qdrant Cloud instead of local:
  QDRANT_URL=https://... QDRANT_API_KEY=... python annotate_golden.py

Requirements:
  pip install qdrant-client
"""

import json
import os
from datetime import datetime
from qdrant_client import QdrantClient

# ─── Configuration ────────────────────────────────────────────────────────────
QDRANT_URL   = os.getenv("QDRANT_URL")        # set for cloud; None = use local
QDRANT_HOST  = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT  = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_KEY   = os.getenv("QDRANT_API_KEY")
COLLECTION   = "locus_items"
OUTPUT_FILE  = "golden_dataset.json"
ANNOTATOR    = "marc"
CANDIDATES   = 20   # how many similar products to show as options
GROUND_TRUTH = 5    # how many you must pick as ground truth
# ──────────────────────────────────────────────────────────────────────────────


def make_client():
    if QDRANT_URL:
        print(f"  Connecting to Qdrant Cloud: {QDRANT_URL}")
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY, check_compatibility=False)
    print(f"  Connecting to local Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)


def load_dataset():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            return json.load(f)
    return []


def save_dataset(data):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  💾  Saved {len(data)} entries → {OUTPUT_FILE}")


def get_point(client, point_id):
    """Retrieve a single Qdrant point by its UUID (with vector)."""
    results = client.retrieve(
        collection_name=COLLECTION,
        ids=[point_id],
        with_payload=True,
        with_vectors=True,
    )
    return results[0] if results else None


def find_similar_products(client, vector, exclude_product_id, top_k=CANDIDATES):
    """
    Search for similar products, deduplicating normal/dark variants.
    Each real product (same product_id) appears only once — the higher-scored one.
    Returns up to top_k unique products.
    """
    # Fetch more than needed to account for deduplication
    raw = client.search(
        collection_name=COLLECTION,
        query_vector=vector,
        limit=top_k * 3,
        with_payload=True,
        score_threshold=0.0,
    )

    seen_products = {}
    for hit in raw:
        pid = hit.payload.get("product_id") or str(hit.id)
        if pid == exclude_product_id:
            continue
        # Keep only the best-scoring variant per product
        if pid not in seen_products or hit.score > seen_products[pid].score:
            seen_products[pid] = hit

    # Sort by score descending and cap at top_k
    unique = sorted(seen_products.values(), key=lambda h: h.score, reverse=True)
    return unique[:top_k]


STALE_SOURCES = {"full_image", "unknown", None, ""}


def box_source_label(box_source):
    """Return a short display tag for box_source quality."""
    if box_source in ("yolo_crop", "title_crop"):
        return f"✓ {box_source}"
    return f"⚠ {box_source or 'unknown'}  ← stale vector"


def print_candidate(rank, hit):
    p   = hit.payload
    url = p.get("image_url", "N/A")
    url_display  = url[:80] + "…" if len(url) > 80 else url
    box_src      = p.get("box_source")
    stale_marker = "  ⚠ STALE" if box_src in STALE_SOURCES else ""
    print(f"  [{rank:2d}]  score={hit.score:.4f}{stale_marker}")
    print(f"        product_id : {p.get('product_id', str(hit.id))}")
    print(f"        Name       : {p.get('name', 'N/A')}")
    print(f"        Category   : {p.get('category_tag', 'N/A')}")
    print(f"        Store      : {p.get('store_name', 'N/A')}")
    print(f"        Crop source: {box_source_label(box_src)}")
    print(f"        Image      : {url_display}")
    print()


def pick_five(candidates):
    while True:
        raw = input(
            f"  Pick {GROUND_TRUTH} numbers separated by commas (e.g. 1,3,5,7,12): "
        ).strip()
        try:
            nums = [int(x.strip()) for x in raw.split(",")]
        except ValueError:
            print("  ⚠️  Numbers and commas only. Try again.\n")
            continue

        if len(nums) != GROUND_TRUTH:
            print(f"  ⚠️  Need exactly {GROUND_TRUTH}, got {len(nums)}. Try again.\n")
            continue

        if any(n < 1 or n > len(candidates) for n in nums):
            print(f"  ⚠️  Numbers must be between 1 and {len(candidates)}. Try again.\n")
            continue

        if len(set(nums)) != GROUND_TRUTH:
            print("  ⚠️  Duplicate numbers. Pick 5 different ones.\n")
            continue

        return [n - 1 for n in nums]   # convert to 0-based indices


def main():
    client  = make_client()
    dataset = load_dataset()

    # Index by product_id (the stable business identifier)
    annotated_product_ids = {entry["query_product_id"] for entry in dataset}

    print()
    print("═" * 64)
    print("  🏷   Locus Golden Dataset Annotator")
    print("═" * 64)
    print(f"  Collection     : {COLLECTION}")
    print(f"  Output file    : {OUTPUT_FILE}")
    print(f"  Already saved  : {len(dataset)} queries")
    print()
    print("  Workflow:")
    print("  1. Paste any Qdrant point UUID for a product you want to test")
    print(f"  2. The tool shows the {CANDIDATES} most similar products")
    print(f"  3. You pick {GROUND_TRUTH} that are the correct ground-truth matches")
    print("  4. Repeat for as many products as you like (20–50 is a solid dataset)")
    print()
    print("  Tip: Open image URLs in your browser to compare products visually.")
    print("  Type 'quit' at any prompt to exit and save.")
    print("═" * 64)

    while True:
        print()
        point_id = input("  Qdrant point UUID: ").strip()

        if point_id.lower() in ("quit", "q", "exit"):
            break

        if not point_id:
            continue

        # Fetch the point
        print("  🔍 Fetching from Qdrant…")
        point = get_point(client, point_id)
        if not point:
            print(f"  ❌ No point '{point_id}' found in '{COLLECTION}'. Check the UUID.")
            continue

        payload    = point.payload
        product_id = payload.get("product_id") or point_id

        # Handle already-annotated product
        if product_id in annotated_product_ids:
            print(f"\n  ⚠️  product_id '{product_id}' is already annotated.")
            choice = input("  Type 'overwrite' to replace it, or Enter to skip: ").strip()
            if choice.lower() != "overwrite":
                continue
            dataset = [e for e in dataset if e["query_product_id"] != product_id]
            annotated_product_ids.discard(product_id)

        query_box_src = payload.get("box_source")
        print()
        print("  ── Query Product ─────────────────────────────────────────")
        print(f"  Point UUID  : {point_id}")
        print(f"  product_id  : {product_id}")
        print(f"  Name        : {payload.get('name', 'N/A')}")
        print(f"  Category    : {payload.get('category_tag', 'N/A')}")
        print(f"  Store       : {payload.get('store_name', 'N/A')}")
        print(f"  Crop source : {box_source_label(query_box_src)}")
        img = payload.get("image_url", "N/A")
        print(f"  Image       : {img[:80]}{'…' if len(img) > 80 else ''}")

        if query_box_src in STALE_SOURCES:
            print()
            print("  ⚠️  WARNING: This product has a STALE vector (box_source =",
                  repr(query_box_src) + ").")
            print("     Its CLIP embedding was taken from the full image, not a")
            print("     tight crop. Consider running reindex_stale.py first, or")
            print("     pick a different query product for more reliable evaluation.")
            choice = input("\n  Continue anyway? (y/n): ").strip().lower()
            if choice != "y":
                continue

        print(f"\n  🔎 Finding {CANDIDATES} most similar unique products…\n")
        candidates = find_similar_products(client, point.vector, product_id)

        if len(candidates) < GROUND_TRUTH:
            print(f"  ❌ Only {len(candidates)} similar products found — need at least {GROUND_TRUTH}.")
            continue

        print(f"  ── {len(candidates)} Candidate Matches ──────────────────────────────────")
        for i, hit in enumerate(candidates):
            print_candidate(i + 1, hit)

        print(f"  Which {GROUND_TRUTH} are the true ground-truth matches for this product?\n")
        chosen = pick_five(candidates)

        # Build the annotation entry
        relevant_product_ids = [
            candidates[i].payload.get("product_id") or str(candidates[i].id)
            for i in chosen
        ]
        relevant_info = [
            {
                "product_id"  : candidates[i].payload.get("product_id") or str(candidates[i].id),
                "name"        : candidates[i].payload.get("name", ""),
                "category_tag": candidates[i].payload.get("category_tag", ""),
                "store_name"  : candidates[i].payload.get("store_name", ""),
                "image_url"   : candidates[i].payload.get("image_url", ""),
                "cosine_sim"  : candidates[i].score,
            }
            for i in chosen
        ]

        entry = {
            "query_product_id"    : product_id,
            "query_point_uuid"    : point_id,
            "query_image_url"     : payload.get("image_url", ""),
            "query_name"          : payload.get("name", ""),
            "query_category_tag"  : payload.get("category_tag", ""),
            "query_store_name"    : payload.get("store_name", ""),
            "query_box_source"    : payload.get("box_source", "unknown"),
            "relevant_product_ids": relevant_product_ids,
            "relevant_info"       : relevant_info,
            "annotated_by"        : ANNOTATOR,
            "created_at"          : datetime.utcnow().isoformat() + "Z",
        }

        dataset.append(entry)
        annotated_product_ids.add(product_id)
        save_dataset(dataset)
        print(f"\n  ✅ Entry saved. Total annotated: {len(dataset)}")

    print()
    print("═" * 64)
    print(f"  Done!  {len(dataset)} queries in {OUTPUT_FILE}")
    print()
    print("  Next step:")
    print("    python evaluate_mlflow.py")
    print("═" * 64)
    print()


if __name__ == "__main__":
    main()
