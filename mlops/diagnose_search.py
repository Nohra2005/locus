#!/usr/bin/env python3
"""
diagnose_search.py — Locus search pipeline diagnostic
======================================================
Answers five questions about why ACS@5 scores are low:

  Q1. Are the relevant product IDs from golden_dataset.json in Qdrant?
  Q2. What category_tag was stored on each relevant item? Matches query_category_tag?
  Q3. What category does /vectorize detect for query images? Matches query_category_tag?
  Q4. Direct cosine similarity between query vector and each stored relevant vector
      (bypasses the category filter entirely).
  Q5. How many relevant items appear in a raw Qdrant search (no filter) vs
      a filtered search (as /search does it)?

Usage (from project root or mlops/):
    python mlops/diagnose_search.py
    # or
    cd mlops && python diagnose_search.py
"""

import base64
import io
import json
import math
import os
import sys
import time

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models

# ── .env loader (no python-dotenv dependency) ─────────────────────────────────
def _load_dotenv(path: str):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


# Search for .env: next to this script's parent, or cwd
_script_dir = os.path.dirname(os.path.abspath(__file__))
for _candidate in [
    os.path.join(_script_dir, "..", ".env"),   # project root when running from mlops/
    os.path.join(os.getcwd(), ".env"),          # cwd
    ".env",
]:
    if os.path.exists(_candidate):
        _load_dotenv(os.path.abspath(_candidate))
        break

# ── Configuration ─────────────────────────────────────────────────────────────
QDRANT_URL      = os.getenv("QDRANT_URL")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")
VISUAL_URL      = os.getenv("VISUAL_URL",  "http://localhost:8001")
COLLECTION      = "locus_items"
TOP_K           = 100      # how many results to fetch from Qdrant for Q5

# Golden dataset: try mlops/golden_dataset.json then cwd
for _p in [
    os.path.join(_script_dir, "golden_dataset.json"),
    os.path.join(os.getcwd(), "golden_dataset.json"),
    os.path.join(os.getcwd(), "mlops", "golden_dataset.json"),
]:
    if os.path.exists(_p):
        GOLDEN_PATH = _p
        break
else:
    GOLDEN_PATH = os.path.join(_script_dir, "golden_dataset.json")
# ──────────────────────────────────────────────────────────────────────────────


def p(msg=""):
    print(msg, flush=True)


def cosine_sim(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if (na and nb) else 0.0


def image_bytes(url: str) -> bytes:
    if url.startswith("data:"):
        _, b64 = url.split(",", 1)
        return base64.b64decode(b64)
    import urllib.request
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Locus-Diagnose/1.0)"}
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read()


def vectorize(img_bytes: bytes):
    """Call /vectorize on the visual engine. Returns (vector, category, confidence)."""
    resp = requests.post(
        f"{VISUAL_URL}/vectorize",
        files={"file": ("query.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        timeout=60,
    )
    resp.raise_for_status()
    d = resp.json()
    return d.get("vector"), d.get("category"), d.get("category_confidence", 0.0)


# ── Fetch ALL golden_dataset items from Qdrant in one scroll pass ─────────────
def fetch_golden_index(client: QdrantClient) -> dict:
    """
    Returns {product_id: {"vector": [...], "category_tag": str, "point_id": str}}
    for every point in locus_items that has store_name="golden_dataset".
    """
    index  = {}
    offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(
                    key="store_name",
                    match=models.MatchValue(value="golden_dataset"),
                )]
            ),
            limit=250,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not batch:
            break
        for pt in batch:
            pid = pt.payload.get("product_id", str(pt.id))
            index[pid] = {
                "vector":       pt.vector,
                "category_tag": pt.payload.get("category_tag", ""),
                "point_id":     str(pt.id),
                "name":         pt.payload.get("name", ""),
            }
        if next_offset is None:
            break
        offset = next_offset
    return index


def main():
    p()
    p("=" * 72)
    p("  LOCUS SEARCH PIPELINE DIAGNOSTIC")
    p("=" * 72)

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    if not QDRANT_URL:
        p("[FATAL] QDRANT_URL is not set — check your .env file.")
        sys.exit(1)

    if not os.path.exists(GOLDEN_PATH):
        p(f"[FATAL] Golden dataset not found at: {GOLDEN_PATH}")
        sys.exit(1)

    p(f"  Qdrant URL   : {QDRANT_URL}")
    p(f"  Visual URL   : {VISUAL_URL}")
    p(f"  Collection   : {COLLECTION}")
    p(f"  Dataset      : {GOLDEN_PATH}")
    p()

    # ── Connect to Qdrant ─────────────────────────────────────────────────────
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
    try:
        info = client.get_collection(COLLECTION)
        p(f"  Collection '{COLLECTION}': {info.vectors_count} vectors total")
    except Exception as e:
        p(f"[FATAL] Cannot reach Qdrant: {e}")
        sys.exit(1)

    # ── Load golden dataset ───────────────────────────────────────────────────
    with open(GOLDEN_PATH) as f:
        dataset = json.load(f)
    p(f"  Golden entries: {len(dataset)}")
    p()

    # ── Batch-fetch all golden_dataset items from Qdrant ─────────────────────
    p("  Fetching all golden_dataset points from Qdrant (single scroll)...")
    t0    = time.time()
    index = fetch_golden_index(client)
    p(f"  → {len(index)} golden_dataset points fetched in {time.time()-t0:.1f}s")
    p()

    # ── Per-query loop ────────────────────────────────────────────────────────
    # Accumulators for summary
    total_relevant     = 0
    total_found        = 0
    total_cat_match    = 0
    total_cat_mismatch = 0

    q1_missing       = []   # (query_name, n_missing, n_total)
    q2_mismatches    = []   # (query_name, pid_short, stored_cat, query_cat)
    q3_mismatches    = []   # (query_name, detected, expected)
    q4_pairs         = []   # (query_name, pid_short, sim)
    q5_rows          = []   # (query_name, raw_k, filt_k, raw_25, filt_25, n_rel)

    for idx, entry in enumerate(dataset):
        query_name   = entry.get("query_name") or f"query_{idx}"
        query_cat    = entry.get("query_category_tag", "")
        relevant_ids = entry.get("relevant_product_ids", [])
        img_url      = entry.get("query_image_url", "")

        p(f"[{idx+1:2d}/{len(dataset)}] {query_name[:60]}")
        p(f"         query_cat='{query_cat}'  |  {len(relevant_ids)} relevant items")

        # ── Q1 & Q2 ──────────────────────────────────────────────────────────
        n_found   = 0
        n_missing = 0
        rel_vecs  = {}   # product_id → vector (only found items)

        for pid in relevant_ids:
            total_relevant += 1
            if pid in index:
                n_found        += 1
                total_found    += 1
                rel_vecs[pid]   = index[pid]["vector"]
                stored_cat      = index[pid]["category_tag"]
                match_sym       = "✓" if stored_cat == query_cat else "✗ MISMATCH"
                p(f"         [Q1/Q2] pid={pid[:8]}...  stored_cat='{stored_cat}'  {match_sym}")
                if stored_cat == query_cat:
                    total_cat_match += 1
                else:
                    total_cat_mismatch += 1
                    q2_mismatches.append((query_name, pid[:8]+"...", stored_cat, query_cat))
            else:
                n_missing += 1
                p(f"         [Q1]    pid={pid[:8]}...  ** NOT IN QDRANT **")

        if n_missing:
            q1_missing.append((query_name, n_missing, len(relevant_ids)))
        p(f"         Q1 → {n_found}/{len(relevant_ids)} found in Qdrant | {n_missing} missing")

        # ── Q3: vectorize query image ─────────────────────────────────────────
        query_vector = None
        try:
            img       = image_bytes(img_url)
            t_v       = time.time()
            query_vector, detected_cat, det_conf_raw = vectorize(img)
            elapsed   = time.time() - t_v
            # category_confidence from /vectorize is the full scores dict in this version
            if isinstance(det_conf_raw, dict):
                det_conf = max(det_conf_raw.values()) if det_conf_raw else 0.0
            else:
                det_conf = float(det_conf_raw) if det_conf_raw is not None else 0.0
            match_sym = "✓" if detected_cat == query_cat else "✗ MISMATCH"
            p(f"         [Q3]    /vectorize → detected='{detected_cat}' "
              f"(conf={det_conf:.2f}) expected='{query_cat}'  {match_sym}  [{elapsed:.1f}s]")
            if detected_cat != query_cat:
                q3_mismatches.append((query_name, detected_cat, query_cat))
        except Exception as e:
            p(f"         [Q3]    /vectorize FAILED: {e}")

        # ── Q4: direct cosine similarity ──────────────────────────────────────
        if query_vector and rel_vecs:
            sims = []
            for pid, vec in rel_vecs.items():
                sim = cosine_sim(query_vector, vec)
                sims.append(sim)
                q4_pairs.append((query_name, pid[:8]+"...", sim))
            avg = sum(sims) / len(sims)
            p(f"         [Q4]    cosine_sim: min={min(sims):.4f}  max={max(sims):.4f}  "
              f"avg={avg:.4f}  (n={len(sims)})")

        # ── Q5: raw vs filtered Qdrant search ─────────────────────────────────
        if query_vector:
            try:
                relevant_set = set(relevant_ids)

                # Raw — no filter
                raw_hits = client.search(
                    collection_name=COLLECTION,
                    query_vector=query_vector,
                    limit=TOP_K,
                    with_payload=True,
                )
                raw_pids    = {r.payload.get("product_id", str(r.id)) for r in raw_hits}
                raw_k       = len(relevant_set & raw_pids)
                raw_25      = len(relevant_set & {
                    r.payload.get("product_id", str(r.id)) for r in raw_hits[:25]
                })

                # Filtered — category_tag = query_cat (mirrors /search)
                filt_filter = (
                    models.Filter(must=[models.FieldCondition(
                        key="category_tag",
                        match=models.MatchValue(value=query_cat),
                    )])
                    if query_cat else None
                )
                filt_hits = client.search(
                    collection_name=COLLECTION,
                    query_vector=query_vector,
                    query_filter=filt_filter,
                    limit=TOP_K,
                    with_payload=True,
                )
                filt_pids = {r.payload.get("product_id", str(r.id)) for r in filt_hits}
                filt_k    = len(relevant_set & filt_pids)
                filt_25   = len(relevant_set & {
                    r.payload.get("product_id", str(r.id)) for r in filt_hits[:25]
                })

                p(f"         [Q5]    raw   top-{TOP_K}={raw_k}/{len(relevant_ids)}  top-25={raw_25}")
                p(f"         [Q5]    filt  top-{TOP_K}={filt_k}/{len(relevant_ids)}  top-25={filt_25}"
                  f"  (filter: category_tag='{query_cat}')")

                q5_rows.append((query_name, raw_k, filt_k, raw_25, filt_25, len(relevant_ids)))

            except Exception as e:
                p(f"         [Q5]    Qdrant search FAILED: {e}")

        p()

    # ═════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═════════════════════════════════════════════════════════════════════════
    p()
    p("=" * 72)
    p("  DIAGNOSTIC SUMMARY")
    p("=" * 72)

    # ── Q1 ───────────────────────────────────────────────────────────────────
    missing_n   = total_relevant - total_found
    missing_pct = missing_n / max(1, total_relevant) * 100
    p()
    p("  Q1 — Relevant items present in Qdrant")
    p(f"       Total relevant items : {total_relevant}")
    p(f"       Found in Qdrant      : {total_found}  ({100-missing_pct:.0f}%)")
    p(f"       MISSING              : {missing_n}  ({missing_pct:.0f}%)")
    if q1_missing:
        p("       Entries with gaps:")
        for name, miss, tot in sorted(q1_missing, key=lambda x: -x[1]):
            p(f"         {miss}/{tot} missing  —  {name[:60]}")

    # ── Q2 ───────────────────────────────────────────────────────────────────
    cat_mm_pct = total_cat_mismatch / max(1, total_found) * 100
    p()
    p("  Q2 — category_tag stored vs query_category_tag")
    p(f"       Category MATCH       : {total_cat_match}  ({100-cat_mm_pct:.0f}%)")
    p(f"       Category MISMATCH    : {total_cat_mismatch}  ({cat_mm_pct:.0f}%)")
    if q2_mismatches:
        p("       Mismatched items (stored ≠ expected):")
        for name, pid, stored, exp in q2_mismatches[:25]:
            p(f"         pid={pid}  stored='{stored}'  expected='{exp}'  — {name[:45]}")

    # ── Q3 ───────────────────────────────────────────────────────────────────
    q3_mm_pct = len(q3_mismatches) / max(1, len(dataset)) * 100
    p()
    p("  Q3 — /vectorize detected category vs query_category_tag")
    p(f"       Correct detections   : {len(dataset)-len(q3_mismatches)}/{len(dataset)}  ({100-q3_mm_pct:.0f}%)")
    p(f"       MISMATCHES           : {len(q3_mismatches)}/{len(dataset)}  ({q3_mm_pct:.0f}%)")
    if q3_mismatches:
        p("       Mismatched queries:")
        for name, det, exp in q3_mismatches:
            p(f"         detected='{det}'  expected='{exp}'  — {name[:50]}")

    # ── Q4 ───────────────────────────────────────────────────────────────────
    p()
    p("  Q4 — Direct cosine similarity (query ↔ relevant, no filter)")
    if q4_pairs:
        all_sims = [s for _, _, s in q4_pairs]
        avg_sim  = sum(all_sims) / len(all_sims)
        p(f"       n pairs              : {len(all_sims)}")
        p(f"       Mean cosine_sim      : {avg_sim:.4f}")
        p(f"       Min / Max            : {min(all_sims):.4f} / {max(all_sims):.4f}")
        # Distribution
        b = {">=0.80": 0, "0.65-0.79": 0, "0.50-0.64": 0, "0.35-0.49": 0, "<0.35": 0}
        for s in all_sims:
            if   s >= 0.80: b[">=0.80"]    += 1
            elif s >= 0.65: b["0.65-0.79"] += 1
            elif s >= 0.50: b["0.50-0.64"] += 1
            elif s >= 0.35: b["0.35-0.49"] += 1
            else:           b["<0.35"]     += 1
        p("       Distribution:")
        for label, cnt in b.items():
            bar = "█" * cnt
            p(f"         {label:<12}  {cnt:3d}  {bar}")
    else:
        p("       No pairs computed (Q3 or Q1 failures prevented this).")

    # ── Q5 ───────────────────────────────────────────────────────────────────
    p()
    p(f"  Q5 — Relevant items in raw vs filtered Qdrant search (top-{TOP_K})")
    if q5_rows:
        sum_raw_k   = sum(r  for _, r,  _, _,  _, _ in q5_rows)
        sum_filt_k  = sum(f  for _, _,  f, _,  _, _ in q5_rows)
        sum_raw_25  = sum(r  for _, _,  _, r,  _, _ in q5_rows)
        sum_filt_25 = sum(f  for _, _,  _, _,  f, _ in q5_rows)
        sum_rel     = sum(n  for _, _,  _, _,  _, n in q5_rows)

        p(f"       Raw   top-{TOP_K}        : {sum_raw_k}/{sum_rel}  ({100*sum_raw_k/max(1,sum_rel):.0f}%)")
        p(f"       Filt  top-{TOP_K}        : {sum_filt_k}/{sum_rel}  ({100*sum_filt_k/max(1,sum_rel):.0f}%)")
        p(f"       Raw   top-25         : {sum_raw_25}/{sum_rel}  ({100*sum_raw_25/max(1,sum_rel):.0f}%)")
        p(f"       Filt  top-25         : {sum_filt_25}/{sum_rel}  ({100*sum_filt_25/max(1,sum_rel):.0f}%)")

        dropped = sum_raw_k - sum_filt_k
        if dropped > 0:
            p(f"       Filter drops         : {dropped} hits ({100*dropped/max(1,sum_raw_k):.0f}% of raw retrievable)")
        p()
        p(f"       Per-query breakdown (raw_top{TOP_K} → filt_top{TOP_K} / total):")
        for name, rk, fk, r25, f25, n in q5_rows:
            delta = rk - fk
            flag  = "  ← filter hurts" if delta > 0 else ""
            p(f"         {rk:2d}→{fk:2d}/{n}  top25: {r25}→{f25}  {name[:45]}{flag}")
    else:
        p("       No Q5 data (vectorize failures prevented this).")

    # ═════════════════════════════════════════════════════════════════════════
    # ROOT CAUSE RANKING
    # ═════════════════════════════════════════════════════════════════════════
    p()
    p("=" * 72)
    p("  ROOT CAUSE RANKING  (most impactful first)")
    p("=" * 72)
    p()

    causes = []

    if total_relevant:
        pct = missing_n / total_relevant
        if pct > 0:
            causes.append((
                pct,
                "CRITICAL" if pct > 0.3 else ("HIGH" if pct > 0.1 else "MEDIUM"),
                f"{pct:.0%} of relevant items are MISSING from Qdrant "
                f"({missing_n}/{total_relevant}) — they were skipped or "
                f"never indexed (likely: YOLO found no box on thumbnail).",
            ))

    if total_found:
        pct = total_cat_mismatch / total_found
        if pct > 0:
            causes.append((
                pct,
                "CRITICAL" if pct > 0.3 else ("HIGH" if pct > 0.1 else "MEDIUM"),
                f"{pct:.0%} of indexed relevant items have a WRONG category_tag "
                f"({total_cat_mismatch}/{total_found}) — the /search category filter "
                f"silently excludes them even though they exist in Qdrant.",
            ))

    if q3_mismatches:
        pct = len(q3_mismatches) / len(dataset)
        causes.append((
            pct,
            "HIGH" if pct > 0.2 else "MEDIUM",
            f"{pct:.0%} of query images are classified to the WRONG category "
            f"by /vectorize ({len(q3_mismatches)}/{len(dataset)}) — the filter "
            f"is applied against the wrong bucket, hiding all correct results.",
        ))

    if q4_pairs:
        all_sims = [s for _, _, s in q4_pairs]
        avg_sim  = sum(all_sims) / len(all_sims)
        low_frac = sum(1 for s in all_sims if s < 0.50) / len(all_sims)
        if avg_sim < 0.55:
            causes.append((
                1 - avg_sim,
                "CRITICAL" if avg_sim < 0.35 else "HIGH",
                f"Mean query↔relevant cosine_sim = {avg_sim:.3f} "
                f"({low_frac:.0%} of pairs < 0.50).  Even without any filter, "
                f"the vectors are too dissimilar to rank relevant items near the top.  "
                f"Likely cause: Google Lens thumbnails (encrypted-tbn* URLs) are "
                f"heavily compressed small images — CLIP embeds the thumbnail, not "
                f"the real product shot.",
            ))

    if q5_rows:
        sum_raw_k  = sum(r for _, r, _, _, _, _ in q5_rows)
        sum_filt_k = sum(f for _, _, f, _, _, _ in q5_rows)
        sum_rel    = sum(n for _, _, _, _, _, n in q5_rows)
        if sum_raw_k > 0:
            filter_drop_pct = (sum_raw_k - sum_filt_k) / sum_raw_k
            if filter_drop_pct > 0.15:
                causes.append((
                    filter_drop_pct,
                    "HIGH" if filter_drop_pct > 0.3 else "MEDIUM",
                    f"Category filter drops {filter_drop_pct:.0%} of items that ARE "
                    f"retrievable by vector alone ({sum_raw_k}→{sum_filt_k}/{sum_rel} "
                    f"in top-{TOP_K}).  This is the combined effect of Q2 + Q3.",
                ))

    if not causes:
        p("  No significant root causes detected — scores may simply reflect")
        p("  genuine embedding distance between query and dataset items.")
    else:
        causes.sort(key=lambda x: -x[0])
        for rank, (_, severity, msg) in enumerate(causes, 1):
            p(f"  #{rank} [{severity}]")
            # wrap at 68 chars
            words  = msg.split()
            line   = "       "
            for w in words:
                if len(line) + len(w) + 1 > 70:
                    p(line)
                    line = "       " + w
                else:
                    line = (line + " " + w).lstrip()
                    line = "       " + line.lstrip()
            if line.strip():
                p(line)
            p()

    p("=" * 72)
    p()


if __name__ == "__main__":
    main()
