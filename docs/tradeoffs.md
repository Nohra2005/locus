# Engineering Tradeoffs

This document records the five primary engineering tradeoffs made during Locus development.
Each entry states what was chosen, what was rejected, and the evidence that justified the decision.

---

## Tradeoff 1 — Latency vs. Accuracy: Approximate HNSW vs. Exhaustive KNN

**Context:** Vector similarity search over the 10k-item Qdrant collection.

| | Chosen | Rejected |
|---|---|---|
| **Approach** | Qdrant HNSW approximate nearest-neighbour (ef=128) | Exhaustive brute-force KNN |
| **Latency** | p95 ~60 ms per query | p95 ~900 ms per query (estimated at 10k scale) |
| **Recall@5** | 0.967 (35-query golden dataset) | ~1.000 (exact, by definition) |

**Decision:** HNSW with ef=128 recovers 96.7 % of the correct top-5 results while reducing query latency by ~15×. The 3.3 % recall gap is acceptable given the user-facing latency benefit and is partially recovered by multi-view indexing (see Tradeoff 5).

**Evidence:** `mlops/evaluate_recall.py --gateway-url http://localhost:8000 --k 5 --mlflow`
Results logged to MLflow experiment `locus_recall_eval`. Current best run: Recall@5 = 0.967, Recall@10 = 0.983.

---

## Tradeoff 2 — Cost vs. Quality: Gemini 2.0 Flash vs. Larger Models

**Context:** Both the Gemini judge (quality scoring) and the attribute tagger call a VLM per search.

| | Chosen | Rejected |
|---|---|---|
| **Model** | Gemini 2.0 Flash (via OpenRouter) | GPT-4o / Gemini 1.5 Pro |
| **Input token cost** | $0.10 / 1M tokens | $5.00 / 1M (GPT-4o), $3.50 / 1M (1.5 Pro) |
| **Judge F1 on golden set** | Adequate (see `mlops/calibrate_judge.py`) | Marginally higher |
| **Monthly LLM cost** | ~$2–5 | ~$80–150 at same volume |

**Decision:** Flash reaches acceptable judge quality at 40–50× lower cost. Total cloud spend: Azure Standard_B2s VM ~$35/month + LLM ~$3–5/month ≈ $38–43/month total. Upgrading to GPT-4o would add ~$100/month with negligible quality gain at current traffic.

**Evidence:** `mlops/calibrate_judge.py` calibration runs; cost model in `README.md` §9.

---

## Tradeoff 3 — Recall vs. Precision: Conditional Category Filter Relaxation

**Context:** Qdrant queries apply a hard `category` filter by default. Some categories share visual embeddings (dress/skirt, bag/accessory) causing the filter to incorrectly discard valid matches.

| | Chosen | Rejected |
|---|---|---|
| **Approach** | Drop the category filter when CLIP confidence is below threshold | Always apply hard category filter |
| **Recall@5 on ambiguous queries** | Recovers ~8 pp drop on dress/skirt and accessory queries | ~8 pp lower recall |
| **Precision@5 impact** | Slight decrease (cross-category results occasionally surface) | Higher precision on unambiguous queries |

**Decision:** The search system serves a discovery use case. Missing a correct result (false negative) is more harmful than occasionally surfacing a close-but-wrong category. Filter relaxation is gated on CLIP confidence to avoid degrading high-confidence queries.

**Evidence:** `gateway/main.py` lines 1658–1682 (dress/skirt), 1645–1657 (accessory). Ablation: hard-filter Recall@5 on ambiguous category subset drops from 0.914 to 0.836.

---

## Tradeoff 4 — Response Latency vs. Attribute Richness: Background Attribute Tagger

**Context:** The attribute tagger (Gemini VLM) extracts 7 structured fields per result image. It takes 3–8 seconds per call.

| | Chosen | Rejected |
|---|---|---|
| **Approach** | Background task — `/search` returns immediately; attributes polled via `/search/{id}/attributes` | Block `/search` until all attributes are returned |
| **`/search` p95 latency** | ~800 ms | ~8 s |
| **Attribute availability** | Delayed ~3–8 s post-response | Synchronous, but search unusable in the interim |

**Decision:** Users expect search results in under a second. The attribute data (colors, style, occasion, pattern) powers the `/refine` endpoint and is a secondary enrichment layer, not a prerequisite to showing results. The polling pattern keeps the core UX fast while attributes load in the background.

**Evidence:** `attribute_tagger/main.py` histogram `locus_tagger_latency_seconds` (buckets 0.5–20 s). Prometheus panel visible in Grafana dashboard.

---

## Tradeoff 5 — Storage vs. Recall: Multi-view Product Indexing

**Context:** Each product has up to 3 images (front, back, lifestyle). They can be indexed as 3 separate vectors or collapsed into 1.

| | Chosen | Rejected |
|---|---|---|
| **Approach** | Index all 3 images as separate vectors; deduplicate by `product_id` at query time | Single canonical image (front) per product |
| **Vector storage** | ~3× more entries in Qdrant | Minimal storage |
| **Recall on lifestyle photography** | Improved — user photos taken at angles match lifestyle shots | Lower on non-front-facing query images |

**Decision:** Qdrant Cloud free tier (1 GB) comfortably holds the current catalog (~10k SKUs × 3 = 30k vectors at 512 dim × float32 = ~60 MB). The storage cost is negligible at this scale; the recall benefit on real-world query images is measurable.

**Evidence:** Multi-view recall logged in `mlops/evaluate_recall.py` runs. Deduplication logic in `gateway/main.py` (best-scoring vector per `product_id` selected, lines 1812–1827).
