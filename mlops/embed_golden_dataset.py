"""
embed_golden_dataset.py
=======================
Converts all external query_image_url entries in golden_dataset.json
to base64 data URIs so the Gemini evaluator never depends on external
network connectivity to fetch query images.

Run once:
    cd c:\dev\locus\mlops
    python embed_golden_dataset.py
"""

import base64
import json
import pathlib
import urllib.request

DATASET_PATH = pathlib.Path(__file__).parent / "golden_dataset.json"
BACKUP_PATH  = DATASET_PATH.with_suffix(".json.bak")


def fetch_image_as_data_uri(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; Locus-Embedder/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw       = resp.read()
        mime      = resp.headers.get_content_type() or "image/jpeg"
        b64       = base64.b64encode(raw).decode("utf-8")
        return f"data:{mime};base64,{b64}"


def main():
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    # Backup original
    BACKUP_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Backup saved to {BACKUP_PATH.name}")

    already_embedded = 0
    converted        = 0
    failed           = []

    for i, entry in enumerate(data):
        url = entry.get("query_image_url", "")
        if url.startswith("data:"):
            already_embedded += 1
            print(f"[{i+1:2d}] already embedded — {entry.get('query_name', '')[:40]}")
            continue

        print(f"[{i+1:2d}] fetching  — {entry.get('query_name', '')[:40]} ...", end=" ", flush=True)
        try:
            entry["query_image_url"] = fetch_image_as_data_uri(url)
            converted += 1
            print(f"ok ({len(entry['query_image_url']) // 1024}KB)")
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(entry.get("query_name", f"entry_{i}"))

    DATASET_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print()
    print(f"Done. {converted} converted, {already_embedded} already embedded, {len(failed)} failed.")
    if failed:
        print(f"Failed entries (URLs kept as-is): {failed}")
    print(f"golden_dataset.json updated.")
    print()
    print("NOTE: The vss_cache.json uses query_image_url as part of the cache key.")
    print("Switching to data URIs means cache entries for the old URLs are now stale.")
    print("Delete vss_cache.json before re-running the evaluator to get fresh scores.")


if __name__ == "__main__":
    main()
