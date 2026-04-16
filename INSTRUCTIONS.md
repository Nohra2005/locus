# Locus — ML System Instruction File
# Claude Execution Guide

## How to use this file
Read this entire document before touching any code. Section 1 defines the
target system. Section 2 evaluates the current codebase against it. Section 3
lists every exact change required, in execution order. Do not skip the audit.

---

## 1. System Vision

### 1.1 Golden Dataset Role
The golden dataset (`mlops/golden_dataset.json`) is the **single source of
truth for all ML experiments**. It has 30 queries, each with exactly 5
annotated relevant product IDs (`relevant_product_ids`).

Every MLflow experiment run **must** evaluate against the golden dataset and
log the system health score. If the score drops below 85, the experiment is
considered a regression and must not be promoted.

### 1.2 System Health Score (SHS) — Primary Metric
```
SHS = (Σ over 30 queries of |relevant_i ∩ top5_i|) / (30 × 5) × 100

Where:
  relevant_i = set of 5 annotated product IDs for query i
  top5_i     = set of product IDs returned in top-5 search results for query i

Target: SHS >= 85
Submittable threshold: SHS >= 85
```

This measures whether ALL relevant items are findable in top 5, not just one.
It is stricter than hit@5. Current system must be evaluated to get baseline SHS.

This metric must be:
- Computed by `evaluate_groq_judge.py` (the new judge script)
- Logged to MLflow as `system_health_score` on every experiment run
- Logged to MLflow as `shs_per_query` (array of 30 per-query scores)

### 1.3 Groq as External Judge (replaces Gemini)
Gemini is dropped — free credits exhausted, not willing to pay.
Groq free tier is used instead: generous RPM, OpenAI-compatible API.

**Groq model:** `meta-llama/llama-4-scout-17b-16e-instruct` (vision capable, free tier)
**Fallback model:** `llama-3.2-11b-vision-preview`

The Groq judge serves two purposes:
1. **Batch evaluation** — scores VSS@5 (Visual Similarity Score) for each
   golden dataset query after a search. Logged to MLflow.
2. **Calibration** — the system prompt includes 3 (query, perfect match)
   examples from the golden dataset so Groq understands what score=10 means
   for Lebanese mall fashion items before scoring the actual results.

The judge does NOT run in the real-time search path (latency too high).
It runs as a background batch evaluator triggered post-experiment.

### 1.4 Ranking Engine Repurposed as Groq Judge Endpoint
The ranking engine (`ranking_engine/`) currently does useless cosine
reranking identical to what Qdrant already does. It must be repurposed.

**New role:** `POST /rank-visual` — accepts a query image URL + list of
candidate image URLs, calls Groq to visually score each candidate, returns
results sorted by Groq's visual similarity score.

This makes the ranking engine a genuine non-trivial IEP for the rubric (T1–T6,
GT3). It is called for quality assessment, not in the live search path.

**Architecture after change:**
```
Real-time search path (fast):
  Query → Gateway → Visual Engine (CLIP embed) → Qdrant (HNSW) → top 25

Quality assessment path (async, demo-able):
  top 25 → Ranking Engine → Groq Vision Judge → quality-scored reranking
```

The gateway calls `/rank-visual` async after returning search results to the
user. Scores are cached in `locus_feedback` or a new `locus_scores` collection
and surfaced in the frontend as "AI quality score" badges.

### 1.5 Latency Requirements
- Real-time `/search`: < 5 seconds end-to-end (Groq NOT in this path)
- `/rank-visual` (quality assessment): acceptable up to 30 seconds (async)
- Groq evaluation batch (30 queries × 5 results): < 10 minutes total
- Rate limit handling: 30 RPM for Groq vision → 2 second sleep between calls

---

## 2. Current Codebase Audit Against Criteria

Read the following files before implementing changes:
- `mlops/evaluate_gemini_judge.py` — to be replaced by Groq version
- `mlops/evaluate_mlflow.py` — already correct, do not modify
- `mlops/golden_dataset.json` — 30 queries, 5 relevant_product_ids each
- `ranking_engine/main.py` + `ranking_engine/ranker.py` — to be repurposed
- `gateway/main.py` — needs async Groq scoring call added
- `monitoring/prometheus.yml` — still scrapes ranking_engine:8002 (stale, fix it)

### 2.1 What is already correct
- `evaluate_mlflow.py`: computes hit@5, precision@5, recall@5, ndcg@5, mrr
  against golden dataset. Logs to MLflow. **Do not change this file.**
- `golden_dataset.json`: 30 queries, 5 relevant IDs each, with image URLs.
  Structure is correct. **Do not change this file.**
- MLflow experiment `locus_search_accuracy`: already exists, has baseline run.
- Prometheus + Grafana: infrastructure exists, dashboards provisioned.
- LoRA retraining pipeline: `mlops/retrain_clip.py` and related scripts.

### 2.2 What is missing or wrong

**MISSING — System Health Score metric:**
`evaluate_mlflow.py` does NOT compute SHS. It computes hit@5 (at least one
relevant in top 5) but not SHS (all relevant items in top 5).
SHS must be added to `evaluate_mlflow.py` and logged as `system_health_score`.

**MISSING — Groq judge:**
`evaluate_gemini_judge.py` uses Gemini API (dead — no credits).
A new `evaluate_groq_judge.py` must be created that:
- Uses Groq API (GROQ_API_KEY from .env)
- Uses golden dataset for few-shot calibration (first 3 entries as examples)
- Computes VSS@5 (visual similarity score) per query
- Computes and logs SHS as part of the same run
- Logs everything to MLflow under `locus_search_accuracy` experiment
- Respects 2s sleep between Groq calls to stay within rate limits

**WRONG — Ranking engine is dead code:**
`ranking_engine/ranker.py` does cosine similarity (identical to Qdrant).
`docker-compose.yml` has it commented out.
Must be replaced with Groq visual judge logic.

**WRONG — Prometheus config scrapes dead ranking engine:**
`monitoring/prometheus.yml` has `ranking_engine:8002` as a scrape target.
This must be updated to still scrape ranking engine but only after it is
re-enabled (or removed if keeping it disabled during transition).

**MISSING — SHS logged in LoRA retraining pipeline:**
`mlops/promote_model.py` evaluates hit@5 and ACS@5 but not SHS.
The promotion condition must include SHS >= 85 as a hard gate.

---

## 3. Implementation Steps — Execute in This Order

### STEP 1: Add System Health Score to evaluate_mlflow.py

File: `mlops/evaluate_mlflow.py`

After computing all existing metrics (hit@5, precision@5, etc.), add:

```python
# System Health Score: fraction of (query, relevant) pairs found in top 5
# SHS = (total relevant items found across all queries) / (30 * 5) * 100
total_relevant_found = 0
total_relevant_possible = 0
shs_per_query = []

for entry, result_ids in zip(golden_queries, all_result_ids):
    relevant = set(entry["relevant_product_ids"])
    found = len(relevant & set(result_ids[:5]))
    total_relevant_found += found
    total_relevant_possible += len(relevant)
    shs_per_query.append(round(found / len(relevant) * 100, 1))

shs = round(total_relevant_found / total_relevant_possible * 100, 1)

mlflow.log_metric("system_health_score", shs)
mlflow.log_metric("shs_queries_passing", sum(1 for s in shs_per_query if s >= 80))
```

The exact location depends on the structure of evaluate_mlflow.py — read it
fully before inserting. The metric must appear in the MLflow run summary.

### STEP 2: Create evaluate_groq_judge.py

File: `mlops/evaluate_groq_judge.py` (new file, replaces evaluate_gemini_judge.py)

Requirements:
- Load GROQ_API_KEY from `.env` (same pattern as evaluate_gemini_judge.py loads GEMINI_API_KEY)
- Use `groq` Python library (add to mlops venv: `pip install groq`)
- Model: try `meta-llama/llama-4-scout-17b-16e-instruct` first, fallback to `llama-3.2-11b-vision-preview`
- Groq API is OpenAI-compatible: `from groq import Groq; client = Groq(api_key=...)`

**Calibration prompt structure:**
```
System prompt:
  "You are a fashion visual similarity judge for a retail search engine.
   You will see a query image and a result image. Score their visual
   similarity from 0 to 10 where:
     10 = identical item (same product, different angle)
      7 = very similar (same style, color, silhouette)
      4 = loosely related (same category, different style)
      1 = unrelated (different category or completely different look)

   Calibration examples from our product database:
   [Include 3 golden dataset entries: show query name + relevant item name
    as text descriptions since we cannot embed images in system prompt]

   Respond with ONLY a JSON object: {\"score\": <0-10>, \"reason\": \"<one sentence>\"}
   Do not include any other text."
```

**Per-query judge flow:**
```
1. Fetch query image bytes (from golden_dataset query_image_url)
2. Call /search on gateway → get top 5 result image URLs
3. For each result image URL:
   a. Fetch image bytes
   b. Call Groq with: [query_image_base64, result_image_base64]
   c. Parse score from JSON response
   d. Sleep 2.0 seconds (rate limit: 30 RPM)
4. VSS@5 = average of 5 scores / 10
5. Also compute SHS for this query (independent of Groq):
   result_product_ids = [r["product_id"] for r in top5_results]
   relevant = set(entry["relevant_product_ids"])
   query_shs = len(relevant & set(result_product_ids)) / len(relevant) * 100
```

**MLflow logging (log into existing experiment `locus_search_accuracy`):**
```python
mlflow.log_metric("vss_at_5", overall_vss)
mlflow.log_metric("system_health_score", overall_shs)
mlflow.log_metric("groq_judge_queries", n_evaluated)
mlflow.log_artifact("groq_judge_results.json")
```

**Cache results** to `mlops/groq_judge_results.json` (same format as
`gemini_judge_results.json`) so the metrics_exporter can pick them up.

**Error handling:**
- If Groq call fails (rate limit, network): log score as None, continue
- If fewer than 20/30 queries succeed: raise error, do not log to MLflow
- Log which queries failed for debugging

### STEP 3: Repurpose ranking_engine/ as Groq Visual Judge Service

Files to change:
- `ranking_engine/ranker.py` — replace cosine logic with Groq judge
- `ranking_engine/main.py` — add `/rank-visual` endpoint, keep `/rank` stub
- `ranking_engine/requirements.txt` — add `groq` library
- `docker-compose.yml` — uncomment ranking_engine, add GROQ_API_KEY env var

**New `ranking_engine/ranker.py`:**
```python
class LocusGroqJudge:
    def __init__(self, api_key: str, model: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model = model
        self.calibration_context = self._build_calibration()  # from golden dataset

    def _build_calibration(self) -> str:
        # Load first 3 golden dataset entries as text calibration examples
        # Returns a string describing perfect matches
        ...

    def score_candidate(self, query_image_b64: str, candidate_image_b64: str) -> dict:
        # Call Groq vision API, return {"score": float, "reason": str}
        ...

    def rank_candidates(self, query_image_url: str,
                        candidate_urls: list[str]) -> list[dict]:
        # Score each candidate, return sorted list with Groq scores
        # Includes 2s sleep between calls for rate limiting
        ...
```

**New endpoint in `ranking_engine/main.py`:**
```python
class RankVisualRequest(BaseModel):
    query_image_url: str
    candidate_image_urls: list[str]  # max 10
    max_candidates: int = 5

@app.post("/rank-visual")
def rank_visual(payload: RankVisualRequest):
    # Calls Groq judge, returns candidates sorted by visual similarity score
    results = judge.rank_candidates(
        payload.query_image_url,
        payload.candidate_image_urls[:payload.max_candidates]
    )
    return {"ranked_candidates": results, "judge_model": judge.model}
```

Keep the old `/rank` endpoint (returns cosine scores) as a fallback stub that
calls the Groq ranker with a note that it now uses visual scoring.

**docker-compose.yml changes:**
```yaml
ranking_engine:
  build: ./ranking_engine
  ports:
    - "8002:8002"
  volumes:
    - ./ranking_engine:/app
    - ./mlops/golden_dataset.json:/app/golden_dataset.json  # for calibration
  environment:
    - GROQ_API_KEY=${GROQ_API_KEY}
```

Add `GROQ_API_KEY=${GROQ_API_KEY}` to gateway's environment block too.

Add `ranking_engine` back to gateway's `depends_on`.

### STEP 4: Add Async Groq Scoring to Gateway /search

File: `gateway/main.py`

After returning search results to the user, fire an async background task
that calls the ranking engine's `/rank-visual` endpoint. Store the scored
results in a new Qdrant collection `locus_scores` or append to
`locus_feedback`.

```python
# At the END of the /search handler, AFTER building `matches`:
async def _score_results_async(query_image_bytes: bytes, matches: list):
    try:
        async with httpx.AsyncClient() as http:
            resp = await http.post(
                f"{RANKING_URL}/rank-visual",
                json={
                    "query_image_url": "...",  # need to handle bytes → temp URL or base64
                    "candidate_image_urls": [m["image_url"] for m in matches[:5]],
                },
                timeout=60.0,
            )
            # Store scores — log to Prometheus or write to locus_scores collection
    except Exception:
        pass  # never block the search response

asyncio.create_task(_score_results_async(image_bytes, matches))
return { ... }  # return immediately, scoring runs in background
```

Note: passing raw image bytes to ranking engine requires either:
  (a) Saving query image to a temp URL (not available without storage)
  (b) Encoding as base64 in the request body (preferred — extend RankVisualRequest)
Use option (b). Update `RankVisualRequest` to accept `query_image_b64: str`.

### STEP 5: Fix prometheus.yml

File: `monitoring/prometheus.yml`

Update the `ranking_engine` scrape job so it is correct after re-enabling:
```yaml
- job_name: 'ranking_engine'
  static_configs:
    - targets: ['ranking_engine:8002']
  scrape_interval: 30s
```

This was previously stale because ranking_engine was disabled. Now that it's
re-enabled, this becomes valid.

### STEP 6: Add SHS Gate to promote_model.py

File: `mlops/promote_model.py`

In the `evaluate_adapter()` function, after computing hit@5 and ACS@5,
also compute SHS using the same golden dataset logic from STEP 1.

In `retrain_clip.py`, the promotion condition must be:
```python
PROMOTION_TOLERANCE = 0.02
shs_gate = new_shs >= 85.0  # hard gate: system must be submittable
quality_gate = new_hit5 >= (baseline_hit5 - PROMOTION_TOLERANCE)
promoted = shs_gate and quality_gate
```

If SHS < 85 but hit@5 improved, still reject — the system is not submittable.

### STEP 7: Update .env.example

File: `.env.example` (or create if missing)

Add:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at: https://console.groq.com/keys

---

## 4. Acceptance Criteria

Before marking any step complete, verify:

| Check | Command / Method | Pass Condition |
|-------|-----------------|----------------|
| SHS logged in MLflow | Run `evaluate_mlflow.py`, check MLflow UI | `system_health_score` appears |
| SHS >= 85 with current model | Check MLflow baseline run | value >= 85.0 |
| Groq judge runs end-to-end | `python evaluate_groq_judge.py` | completes 30 queries, logs VSS@5 |
| Groq respects rate limit | Check logs | 2s sleep between Groq calls |
| Ranking engine starts | `docker compose up ranking_engine` | health check passes |
| `/rank-visual` responds | `curl -X POST .../rank-visual` | returns ranked_candidates |
| Groq calibration in prompt | Check judge system prompt | golden dataset examples present |
| prometheus.yml valid | Check Prometheus targets page | ranking_engine shows as UP |
| Promotion gate includes SHS | Read `retrain_clip.py` | `new_shs >= 85` in condition |

---

## 5. File Change Summary

| File | Action |
|------|--------|
| `mlops/evaluate_mlflow.py` | ADD: system_health_score metric computation and logging |
| `mlops/evaluate_groq_judge.py` | CREATE: full replacement for evaluate_gemini_judge.py |
| `mlops/promote_model.py` | MODIFY: add SHS computation, add SHS >= 85 gate |
| `mlops/retrain_clip.py` | MODIFY: promotion condition adds SHS gate |
| `ranking_engine/ranker.py` | REPLACE: Groq visual judge instead of cosine similarity |
| `ranking_engine/main.py` | ADD: /rank-visual endpoint; keep /rank as stub |
| `ranking_engine/requirements.txt` | ADD: `groq>=0.8.0` |
| `gateway/main.py` | ADD: async Groq scoring background task in /search |
| `docker-compose.yml` | ENABLE: ranking_engine; ADD: GROQ_API_KEY env var |
| `monitoring/prometheus.yml` | FIX: ranking_engine scrape target now valid |
| `.env.example` | ADD: GROQ_API_KEY placeholder |

Do NOT modify:
- `mlops/evaluate_mlflow.py` structure (only add SHS metric)
- `mlops/golden_dataset.json`
- `mlops/golden_dataset.json.bak`
- Any LoRA retraining files (`lora_trainer.py`, `build_training_pairs.py`, etc.)
- `visual_engine/` files except where already modified for LoRA

---

## 6. Notes on Groq API Usage

```python
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {"role": "system", "content": CALIBRATION_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{query_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{result_b64}"}},
            {"type": "text", "text": "Score the visual similarity of these two fashion items."}
        ]}
    ],
    max_tokens=100,
    temperature=0.1,  # low temperature for consistent scoring
)

raw = response.choices[0].message.content
# Parse JSON from raw — use regex fallback if JSON parse fails
```

Rate limits (free tier as of 2026):
- Vision models: 30 RPM, 7000 TPM
- Sleep 2.0 seconds between calls to stay safe

If `meta-llama/llama-4-scout-17b-16e-instruct` is unavailable, fall back to
`llama-3.2-11b-vision-preview`. Both support image inputs.
