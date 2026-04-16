# Locus — Claude Instruction File
**Feed this file to Claude at the start of a new session to resume work.**

---

## 0. Project Context

Locus is a visual search engine for Lebanese mall fashion. Shoppers photograph clothing items and find similar products in nearby stores. The system runs as a Docker Compose stack:

| Service | Port | Role |
|---|---|---|
| Gateway (EEP) | 8000 | FastAPI orchestrator |
| Visual Engine (IEP 1) | 8001 | CLIP ViT-B/16 (`patrickjohncyh/fashion-clip`) + YOLO detection |
| Ranking Engine (IEP 2) | 8002 | Currently dead code — see Section 4C |
| Qdrant | cloud | Vector store, cosine similarity, 3 collections |
| MLflow | 5000 | Experiment tracking |
| Prometheus + Grafana | 9090 / 3000 | Observability |

**Working directory:** `locus/` (all file paths in this document are relative to it).

---

## 1. System Health Metric — Canonical Definition

> **System Health Score (SHS) = (number of (query, ground_truth) pairs where the ground_truth product appears in the top-5 search results) / 150 × 100**

**How to compute it:**
- The golden dataset (`mlops/golden_dataset.json`) has 30 queries, each with exactly 5 `relevant_product_ids`.
- Total pairs = 30 × 5 = 150.
- For each query: send the `query_image_url` to `POST /search`, get the top-5 `product_id` values returned.
- Count how many of the 5 relevant products appear in those top-5 results.
- Sum across all 30 queries, divide by 150, multiply by 100.

**This is mathematically identical to precision@5 = recall@5** (both equal because k=5 and n_relevant=5 for every query).

**Current baseline:** SHS = 73.3 (precision@5 = 0.733 from MLflow run `fashion_clip_baseline`).

**Submission threshold:** SHS must be ≥ 85 for the project to be submittable.

**Note:** hit@5 (currently 0.967) is NOT the system health metric — it only checks if at least 1 of 5 relevant items appears. SHS is stricter: it measures all 5.

---

## 2. Pre-Work Audit — Run This Before Making Any Changes

Before touching any code, Claude must:

### 2A. Run the baseline evaluation and confirm SHS
```bash
cd mlops/
GATEWAY_URL=http://localhost:8000 MLFLOW_TRACKING_URI=http://localhost:5000 \
  python evaluate_mlflow.py
```
Confirm that MLflow logs `precision_at_5` ≥ 0.73. If not, the golden dataset or Qdrant collection has changed — stop and investigate before proceeding.

### 2B. Check evaluate_mlflow.py computes SHS correctly
File: `mlops/evaluate_mlflow.py`

Verify it:
- Loads all 30 entries from `golden_dataset.json`
- Sends each `query_image_url` to `POST /search`
- Compares returned `product_id` values against `relevant_product_ids`
- Logs `precision_at_5` to MLflow (this is SHS/100)

If `precision_at_5` is not being logged, add it. The formula:
```python
precision_at_5 = hits_in_top5 / len(relevant_ids)   # per query, where len = 5
# average across all 30 queries → this is the SHS/100
```

### 2C. Verify all Docker services are up
```bash
docker compose ps
curl -sf http://localhost:8000/ && echo "Gateway OK"
curl -sf http://localhost:8001/ && echo "Visual Engine OK"
curl -sf http://localhost:5000/ && echo "MLflow OK"
```

### 2D. Check Groq API key is available
The new judge requires a Groq API key (free tier). Verify `GROQ_API_KEY` is in the `.env` file. If not, prompt the user to add it before continuing.

---

## 3. Required Change 1 — Replace Gemini Judge with Groq Judge

### Why
The Gemini free credit limit is exhausted. Groq offers a much more generous free tier with faster inference (~200ms per request on dedicated chips). We use Groq's vision model to judge visual similarity between query images and search results.

### Groq model to use
`meta-llama/llama-4-scout-17b-16e-instruct` — supports vision input (base64 images), fast, free tier.

### What to build
New file: `mlops/evaluate_groq_judge.py`

This replaces `mlops/evaluate_gemini_judge.py`. Keep the old file — do not delete it.

**Behavior:**
1. Load `golden_dataset.json` (30 queries).
2. For each query, call `POST /search` and get top-5 results.
3. For each result, call Groq with the query image + result image and ask it to score visual similarity 0–10.
4. Compute VSS@5 = average score across top-5, divided by 10 (normalized to 0–1).
5. Log all scores to MLflow under experiment `locus_search_accuracy`, metric name `vss_at_5`.

**Latency constraint:** The judge must complete all 30 queries in under 5 minutes total. To achieve this:
- Batch all 5 result images for a query into a single Groq request (one call per query, not 5).
- The prompt asks Groq to score all 5 at once and return a JSON array of scores.
- Add `time.sleep(0.5)` between queries to stay within Groq rate limits.

### Groq prompt structure (use this exactly)

```python
SYSTEM_PROMPT = """You are a fashion visual similarity judge for a retail search engine.
You will be shown a query image (what the shopper is looking for) and up to 5 result images
returned by the search engine.

Score each result from 0 to 10 based on visual similarity to the query:
  10 = identical item (same product, slightly different angle)
   8 = very similar (same style, color, silhouette)
   6 = somewhat similar (same category, similar style but different color or cut)
   4 = loosely related (same category, clearly different style)
   2 = different category or completely different look
   0 = unrelated

Respond ONLY with a JSON array of integers, one score per result image, in order.
Example for 5 results: [8, 6, 3, 9, 4]"""
```

### Golden dataset calibration (few-shot in the prompt)

Before the query images, include 2 calibration examples from the golden dataset. This anchors Groq's scoring scale to real examples of what "perfect" and "poor" matches look like.

```python
CALIBRATION_EXAMPLES = """
CALIBRATION — Examples of correct scoring:
[Show golden_dataset entry 0: query image + its first relevant_product image → expected score: 10]
[Show golden_dataset entry 0: query image + an unrelated product → expected score: 1]
Use these as reference when scoring the actual query below.
"""
```

**Implementation note:** Download the first golden dataset query image and its first `relevant_product_ids` image once at startup. Encode both as base64. Include them in every Groq request as part of the system context. This is the calibration anchor.

### Groq client setup

```python
from groq import Groq
import base64, httpx

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def _encode_image_url(url: str) -> str:
    """Download image and return base64 string."""
    resp = httpx.get(url, timeout=15, follow_redirects=True,
                     headers={"User-Agent": "Mozilla/5.0"})
    return base64.b64encode(resp.content).decode("utf-8")

def judge_results(query_url: str, result_urls: list[str]) -> list[float]:
    """Score up to 5 result images against a query. Returns scores 0–10."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    content = [{"type": "text", "text": "Query image (what the shopper wants):"},
               {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode_image_url(query_url)}"}}]
    
    for i, url in enumerate(result_urls[:5]):
        content.append({"type": "text", "text": f"Result {i+1}:"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode_image_url(url)}"}})
    
    content.append({"type": "text", "text": "Score each result (JSON array only):"})
    messages.append({"role": "user", "content": content})
    
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages,
        max_tokens=50,
        temperature=0.0,
    )
    
    import json as _json
    raw = response.choices[0].message.content.strip()
    scores = _json.loads(raw)
    return [float(s) for s in scores]
```

### MLflow logging

After running all 30 queries, log:
- `vss_at_5`: mean Groq score / 10 across all queries (0–1)
- `groq_score_mean`: mean raw score (0–10) for human readability
- `groq_score_min` / `groq_score_max`: distribution
- Per-query scores as a JSON artifact

---

## 4. Required Change 2 — Repurpose Ranking Engine as Groq Re-Ranker

### Why
The ranking engine (`ranking_engine/`) is currently dead code. It runs cosine similarity which Qdrant already does. Replacing its logic with Groq-based visual re-ranking makes it a genuinely non-trivial IEP (required for rubric T1–T6).

### Architecture after this change

```
POST /search (Gateway)
  → Visual Engine /vectorize          → 512-dim CLIP vector
  → Qdrant HNSW search (top-25)       → approximate candidates
  → Ranking Engine /rank (NEW)         → Groq judges top-10 visually
  → Gateway deduplication              → top-5 returned to shopper
```

The Groq judge is called only when the search has a query image available. The ranking engine receives the query image URL and candidate image URLs, calls Groq to score each, and returns re-ranked results with Groq scores attached.

### Changes to `ranking_engine/ranker.py`

Replace the entire cosine similarity implementation with:

```python
import os, json, time, base64
import httpx
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_CANDIDATES_TO_JUDGE = 10   # judge only top-10 to keep latency under 2s

class LocusRanker:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

    def predict(self, query_image_url: str, candidate_image_urls: list[str],
                qdrant_scores: list[float]) -> list[dict]:
        """
        Re-rank candidates using Groq visual judge.
        Falls back to Qdrant cosine scores if Groq is unavailable.

        Args:
            query_image_url:    URL of the shopper's query image
            candidate_image_urls: URLs of the top-N candidate product images
            qdrant_scores:      Cosine similarity scores from Qdrant (same order)

        Returns:
            List of {index, score, groq_score, source} sorted by score descending.
            source is "groq" or "qdrant_fallback".
        """
        if not self.client or not query_image_url:
            return self._qdrant_fallback(qdrant_scores)

        # Only judge top-MAX_CANDIDATES_TO_JUDGE by Qdrant score
        n = min(MAX_CANDIDATES_TO_JUDGE, len(candidate_image_urls))
        
        try:
            groq_scores = self._judge(query_image_url, candidate_image_urls[:n])
            results = []
            for i, (gs, qs) in enumerate(zip(groq_scores, qdrant_scores[:n])):
                # Weighted combination: 70% Groq visual score + 30% Qdrant cosine
                combined = 0.7 * (gs / 10.0) + 0.3 * qs
                results.append({"index": i, "score": round(combined, 4),
                                 "groq_score": gs, "source": "groq"})
            # Append remaining candidates (beyond top-10) with Qdrant scores only
            for i in range(n, len(qdrant_scores)):
                results.append({"index": i, "score": round(0.3 * qdrant_scores[i], 4),
                                 "groq_score": None, "source": "qdrant_only"})
            return sorted(results, key=lambda x: x["score"], reverse=True)
        except Exception as e:
            print(f"[RANKER] Groq failed ({e}), falling back to Qdrant scores")
            return self._qdrant_fallback(qdrant_scores)

    def _qdrant_fallback(self, qdrant_scores):
        return [{"index": i, "score": round(s, 4), "groq_score": None,
                 "source": "qdrant_fallback"}
                for i, s in enumerate(qdrant_scores)]

    def _judge(self, query_url: str, candidate_urls: list[str]) -> list[float]:
        # [Use the same judge_results() implementation from Section 3]
        # Returns list of raw scores 0-10
        ...
```

### Changes to `ranking_engine/main.py`

Update the `RankRequest` model to accept image URLs and Qdrant scores:

```python
class RankRequest(BaseModel):
    query_image_url:      str
    candidate_image_urls: List[str]
    qdrant_scores:        List[float]
```

### Changes to `gateway/main.py`

In the `/search` handler, after the Qdrant search:
1. Pass `with_vectors=False` (we no longer need vectors — we use image URLs instead).
2. Build `candidate_image_urls` from `hit.payload.get("image_url")`.
3. Call `POST http://ranking_engine:8002/rank` with the new payload.
4. Use the returned ranked indices to reorder results.

Add graceful fallback: if ranking engine is unreachable, continue with Qdrant order.

**Latency budget:** The Groq judge must complete within 3 seconds for a /search call to remain responsive. With batched scoring of top-10 candidates in a single Groq call, this is achievable. Log ranking latency as a Prometheus histogram metric `locus_ranking_latency_seconds`.

### Re-enable ranking engine in docker-compose.yml

Uncomment the `ranking_engine` service block and add `GROQ_API_KEY=${GROQ_API_KEY}` to its environment. Add it back to gateway's `depends_on`.

---

## 5. Required Change 3 — Wire SHS into MLflow as the Primary Metric

### What to change in `mlops/evaluate_mlflow.py`

1. Rename the MLflow metric `precision_at_5` to `system_health_score` (multiply by 100 before logging so the dashboard shows 0–100, not 0–1).
2. Add a clear PASS/FAIL log line:
   ```python
   status = "PASS ✅" if shs >= 85 else "FAIL ❌ (target: 85)"
   print(f"\n  System Health Score: {shs:.1f}/100  {status}")
   mlflow.log_metric("system_health_score", shs)
   mlflow.set_tag("shs_pass", str(shs >= 85))
   ```
3. After running `evaluate_groq_judge.py`, also log `vss_at_5` in the same MLflow run.

### Experiment naming convention

Every time the architecture changes (LoRA training, Groq integration, etc.), run a new MLflow experiment with a descriptive run name:
- `fashion_clip_baseline` — original CLIP, no LoRA
- `fashion_clip_lora_v1` — after first LoRA fine-tuning
- `fashion_clip_lora_groq_rerank` — LoRA + Groq re-ranking

The SHS from each run must be logged so improvement is traceable.

---

## 6. Required Change 4 — Add GROQ_API_KEY to Environment

### `.env` file (root of `locus/`)
Add:
```
GROQ_API_KEY=your_groq_api_key_here
```

### `docker-compose.yml`
Add `GROQ_API_KEY=${GROQ_API_KEY}` to:
- `gateway` service environment
- `ranking_engine` service environment
- `mlops_retrain` service environment

### `ranking_engine/requirements.txt`
Create/update with:
```
groq>=0.7.0
httpx>=0.27.0
fastapi
uvicorn
prometheus-fastapi-instrumentator
```

---

## 7. Execution Order for Claude

When this file is fed to Claude, execute steps in this exact order:

**Step 1 — Audit (read-only, no changes)**
- Read `mlops/evaluate_mlflow.py` and verify SHS metric formula.
- Check `ranking_engine/ranker.py` to confirm it is still dead cosine similarity code.
- Check `docker-compose.yml` for `GROQ_API_KEY` presence.
- Check `mlops/evaluate_gemini_judge.py` to understand what to replace.
- Run the baseline evaluation and record current SHS.

**Step 2 — Build Groq judge**
- Create `mlops/evaluate_groq_judge.py` per Section 3.
- Test locally: `GROQ_API_KEY=xxx python evaluate_groq_judge.py --dry-run` (limit to 3 queries).
- Confirm it returns valid VSS@5 scores without errors.

**Step 3 — Update evaluate_mlflow.py**
- Add `system_health_score` metric per Section 5.
- Run a new MLflow experiment to confirm SHS logs correctly.

**Step 4 — Build Groq ranking engine**
- Replace `ranking_engine/ranker.py` per Section 4.
- Update `ranking_engine/main.py` per Section 4.
- Add `ranking_engine/requirements.txt`.
- Wire gateway `/search` to call ranking engine per Section 4.
- Re-enable ranking engine in `docker-compose.yml`.

**Step 5 — Environment**
- Verify `.env` has `GROQ_API_KEY`.
- Verify all services have it in `docker-compose.yml`.

**Step 6 — Integration test**
- `docker compose up --build`
- Run one search query end-to-end and confirm Groq scores appear in the response.
- Check ranking engine logs show `[RANKER] Groq scored N candidates`.

**Step 7 — Final SHS measurement**
- Run `evaluate_mlflow.py` with run name `fashion_clip_groq_rerank`.
- Confirm `system_health_score` ≥ 85 in MLflow.
- If SHS < 85, proceed to LoRA fine-tuning (see `mlops/retrain_clip.py`).

---

## 8. If SHS < 85 After Groq Integration

The Groq re-ranker improves result ordering but does NOT change the underlying CLIP embeddings. If SHS is still below 85 after Groq integration, run the LoRA fine-tuning pipeline:

```bash
# Force a training run immediately (skips the 50-pair threshold check)
docker compose exec mlops_retrain python retrain_clip.py --force
```

After promotion, re-run `evaluate_mlflow.py` with run name `fashion_clip_lora_v1` and check SHS again.

The target improvement path:
- Baseline:             SHS ≈ 73.3
- After Groq re-rank:  SHS ≈ 78–82 (better ordering, not better embeddings)
- After LoRA v1:       SHS ≈ 82–88 (better embeddings)
- Combined:            SHS ≈ 85–92 (target met)

---

## 9. Files Claude Must NOT Modify Unless Explicitly Instructed

- `mlops/golden_dataset.json` — ground truth, immutable
- `mlops/mlruns/` — MLflow artifact history, do not delete
- `visual_engine/clip_labels.py` — canonical category list, changes break indexing
- Any `.env` file — show the user what to add, do not write API keys yourself
- `k8s/` — Kubernetes manifests, already satisfy S4 rubric requirement

---

## 10. Key File Paths Reference

```
locus/
├── gateway/main.py                  # EEP orchestrator — add /trigger-retrain, /reindex, update /search
├── visual_engine/vectorizer.py      # _clip_embed() uses vision_model + visual_projection
├── visual_engine/main.py            # Add /reload-adapter endpoint
├── ranking_engine/ranker.py         # REPLACE with Groq judge
├── ranking_engine/main.py           # Update RankRequest schema
├── mlops/
│   ├── evaluate_mlflow.py           # ADD system_health_score metric
│   ├── evaluate_gemini_judge.py     # KEEP but superseded by evaluate_groq_judge.py
│   ├── evaluate_groq_judge.py       # CREATE — Groq visual similarity judge
│   ├── golden_dataset.json          # 30 queries × 5 relevant products = 150 pairs
│   ├── build_training_pairs.py      # LoRA data pipeline (already built)
│   ├── lora_trainer.py              # LoRA training loop (already built)
│   ├── retrain_clip.py              # Pipeline orchestrator (already built)
│   └── promote_model.py             # Evaluate + promote adapter (already built)
├── docker-compose.yml               # Re-enable ranking_engine, add GROQ_API_KEY
└── .env                             # Add GROQ_API_KEY (user must supply key)
```

---

## 11. Rubric Satisfaction Checklist

Before submitting, verify each rubric item is demonstrably met:

| Code | Item | Evidence needed |
|---|---|---|
| GT1 | Live demo works | End-to-end search returns results with Groq scores |
| GT2 | Cloud deployment | Public URL reachable (deploy to Railway/Render/AWS) |
| GT3 | EEP + 2 IEPs | Gateway + Visual Engine + Ranking Engine (Groq judge) |
| GT4 | All deliverables | Repo, docs, deployment link, demo video |
| T1–T6 | Engineering tradeoffs | Document: HNSW vs exact, LoRA vs full fine-tune, Groq vs local re-ranker |
| S4 | Kubernetes | `k8s/deployment.yaml`, `k8s/service.yaml`, `k8s/ingress.yaml` (already exist) |
| Q1 | Tests | `tests/test_smoke.py` (already exists) |
| M2 | Experiment tracking | MLflow with `system_health_score`, `vss_at_5` per run |
| M3 | Prometheus + Grafana | `locus_ranking_latency_seconds` histogram, existing dashboard |
| M1 | Automated pipeline | `mlops/retrain_clip.py` with `--schedule 48` in docker-compose |

**Minimum SHS to submit: 85.0**
