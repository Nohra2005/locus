"""
bulk_index.py — Scrape a store and index all products into Locus.

Usage:
    python bulk_index.py <store_url> <store_name> [address]   # start / resume
    python bulk_index.py --reset                               # clear checkpoint and start fresh

Supports pause/resume: just kill the process at any time.
On next run it will pick up from the last completed batch.
Checkpoint: bulk_index_checkpoint.json
Products cache: bulk_index_products.json
"""
import sys
import json
import time
import os
import requests

GATEWAY     = "http://localhost:8000"
CHUNK       = 10
CHECKPOINT  = os.path.join(os.path.dirname(__file__), "bulk_index_checkpoint.json")
PRODUCTS_FILE = os.path.join(os.path.dirname(__file__), "bulk_index_products.json")

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(os.path.dirname(__file__), "bulk_index.log"), "a") as f:
        f.write(line + "\n")

def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            return json.load(f)
    return None

def save_checkpoint(data):
    with open(CHECKPOINT, "w") as f:
        json.dump(data, f)

def scrape(url):
    log(f"Scraping {url} ...")
    r = requests.post(f"{GATEWAY}/scrape", json={"url": url, "max_products": 0}, timeout=120)
    r.raise_for_status()
    d = r.json()
    log(f"Strategy: {d.get('strategy')}  |  Found: {d.get('total_found')} products")
    return d.get("products", [])

def index_all(products, store_name, address, start_batch=0, prev_success=0, prev_skipped=0, prev_failed=0):
    total      = len(products)
    success    = prev_success
    skipped    = prev_skipped
    failed     = prev_failed
    total_bat  = (total + CHUNK - 1) // CHUNK

    for batch_num in range(start_batch + 1, total_bat + 1):
        start = (batch_num - 1) * CHUNK
        chunk = products[start:start + CHUNK]

        items = [
            {
                "name":       p.get("name", "Product"),
                "store":      store_name,
                "mall":       address,
                "image_urls": p.get("image_urls") or ([p["image_url"]] if p.get("image_url") else []),
                "image_url":  p.get("image_url", ""),
                "price":      p.get("price", ""),
                "category":   p.get("category", ""),
            }
            for p in chunk
        ]

        try:
            resp = requests.post(f"{GATEWAY}/add-bulk-batch", json={"items": items}, timeout=300)
            resp.raise_for_status()
            data     = resp.json()
            success += data.get("success", 0)
            skipped += data.get("skipped", 0)
            failed  += len(data.get("failed", []))
        except Exception as e:
            failed += len(chunk)
            log(f"Batch {batch_num}/{total_bat} ERROR: {e}")

        log(f"Batch {batch_num}/{total_bat} | ok={success} skip={skipped} fail={failed}")
        save_checkpoint({
            "url": products[0].get("_source_url", ""),
            "store_name": store_name,
            "address": address,
            "last_completed_batch": batch_num,
            "success": success,
            "skipped": skipped,
            "failed": failed,
            "total": total,
        })

    log(f"Done — success={success}  skipped={skipped}  failed={failed}  total={total}")
    # Clean up checkpoint on successful completion
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)
    if os.path.exists(PRODUCTS_FILE):
        os.remove(PRODUCTS_FILE)

if __name__ == "__main__":
    if "--reset" in sys.argv:
        for f in [CHECKPOINT, PRODUCTS_FILE]:
            if os.path.exists(f):
                os.remove(f)
        print("Checkpoint cleared.")
        sys.exit(0)

    cp = load_checkpoint()

    if cp and os.path.exists(PRODUCTS_FILE):
        # Resume from checkpoint
        with open(PRODUCTS_FILE) as f:
            products = json.load(f)
        store_name  = cp["store_name"]
        address     = cp["address"]
        start_batch = cp["last_completed_batch"]
        log(f"Resuming from batch {start_batch + 1}/{cp['total'] // CHUNK + 1} "
            f"(already indexed: ok={cp['success']} skip={cp['skipped']} fail={cp['failed']})")
        index_all(products, store_name, address,
                  start_batch=start_batch,
                  prev_success=cp["success"],
                  prev_skipped=cp["skipped"],
                  prev_failed=cp["failed"])
    else:
        # Fresh start
        url        = sys.argv[1] if len(sys.argv) > 1 else "https://www.forever21.com/collections/all"
        store_name = sys.argv[2] if len(sys.argv) > 2 else "Forever21"
        address    = sys.argv[3] if len(sys.argv) > 3 else ""

        with open(os.path.join(os.path.dirname(__file__), "bulk_index.log"), "a") as f:
            f.write(f"\n=== bulk_index started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

        products = scrape(url)
        if not products:
            log("No products found. Exiting.")
            sys.exit(1)

        # Cache products so resume doesn't need to re-scrape
        with open(PRODUCTS_FILE, "w") as f:
            json.dump(products, f)

        index_all(products, store_name, address)
