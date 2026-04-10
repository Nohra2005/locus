# Locus — Planned Changes Spec

> **For Claude Code:** This document describes three precise, self-contained changes to implement.
> Read the full spec for each change before writing any code. Do not infer intent — follow
> these instructions exactly. Each change has a clear scope, file list, and acceptance criteria.

---

## Overview of Changes

| # | Change | Scope |
|---|--------|-------|
| 1 | Add `source` field to `/feedback` endpoint | `gateway/main.py` |
| 2 | Groq live async judge — fires per search, feeds `/feedback` | `gateway/main.py`, new `gateway/judge.py`, `.env.example` |
| 3 | `is_golden` tag on golden items + dev toggle in search | `gateway/main.py`, `mlops/tools/build_golden_from_lens.py` |
| 4 | Replace VSS@5 with Recall@5 as primary eval metric | new `mlops/evaluate_recall.py` |

---

## Change 1 — Add `source` field to `/feedback` endpoint

### Why
User feedback and auto-judge feedback will coexist in `locus_feedback`. They must be
distinguishable so training pipelines can weight them differently and audits can separate them.

### What to change

**File:** `gateway/main.py`

1. In the `FeedbackRequest` Pydantic model, add one new optional field:
   ```python
   source: str = "user"   # "user" | "auto_judge"
   ```

2. In the `/feedback` POST handler, pass `source` through to the Qdrant payload dict
   that gets stored in `locus_feedback`. It should sit alongside the existing fields
   (`rating`, `training_signal`, `weight`, `timestamp`, etc.).

3. Add `source` as an indexed keyword payload field on the `locus_feedback` Qdrant
   collection (add it to the `create_payload_index` calls that already exist for
   `training_signal`, `category`, etc.).

4. In the `GET /feedback` handler, add an optional query param `source: str = None`
   that filters results by source when provided.

### Acceptance criteria
- `POST /feedback` with no `source` field stores `"user"` in the payload
- `POST /feedback` with `source="auto_judge"` stores `"auto_judge"`
- `GET /feedback?source=auto_judge` returns only auto-judge records
- `GET /feedback?source=user` returns only user records
- `GET /feedback` with no source param returns all records (existing behaviour)

---

## Change 2 — Groq Live Async Judge

### Why
We want to auto-generate retraining signal on every real user search, without any
user interaction. The judge compares each returned result image against the query image,
scores similarity 0.00–1.00, converts to a 1–5 star rating, and calls `POST /feedback`
with `source="auto_judge"`. This must be **fire-and-forget** — it must not block the
search response to the user.

We use Groq (llama-3.2-11b-vision-preview) instead of Gemini because Groq is more
generous on rate limits and sufficiently capable for pairwise image comparison.

### Scope

- **New file:** `gateway/judge.py` — the entire judge logic lives here
- **Modified file:** `gateway/main.py` — wires the judge into the search endpoint as a background task
- **Modified file:** `.env.example` — add `GROQ_API_KEY` entry

### Implementation — `gateway/judge.py`

Create this file from scratch. It must contain:

#### Constants
```python
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
JUDGE_PROMPT = (
    "You are a fashion visual similarity expert.\n"
    "You will be shown two clothing images: first the QUERY image, then a RESULT image.\n"
    "Rate how visually similar the result is to the query.\n"
    "Give a score from 0.00 to 1.00 using exactly two decimal places.\n"
    "Anchor points:\n"
    "  1.00 = identical item\n"
    "  0.80 = very similar (same style, colour, silhouette)\n"
    "  0.60 = similar style (same category, close design)\n"
    "  0.40 = same category only\n"
    "  0.20 = loosely related\n"
    "  0.00 = unrelated\n"
    "Respond with ONLY the numeric score. No explanation."
)
```

#### Score → star rating mapping
```
0.80 – 1.00  →  5 stars
0.60 – 0.79  →  4 stars
0.40 – 0.59  →  3 stars  (neutral in training — still stored)
0.20 – 0.39  →  2 stars
0.00 – 0.19  →  1 star
```

#### Function: `score_to_stars(score: float) -> int`
Implements the mapping above. Returns int 1–5.

#### Function: `fetch_image_as_base64(url: str) -> str`
Downloads an image from a URL and returns it as a base64-encoded JPEG string.
Must handle connection errors gracefully — return `None` on failure, do not raise.

#### Function: `judge_pair(query_image_b64: str, result_image_url: str, groq_api_key: str) -> float | None`
- Downloads `result_image_url` via `fetch_image_as_base64`
- Calls the Groq chat completions API with two images in the message:
  - First image: the query (already base64, passed in as param)
  - Second image: the result (base64 from fetch)
- Parses the float from the response
- Returns the float score, or `None` on any error (network, parse, API error)
- Must not raise exceptions — log errors and return `None`

#### Function: `run_judge(query_image_bytes: bytes, results: list[dict], gateway_base_url: str, groq_api_key: str) -> None`
This is the main entry point called as a background task.

- Encode `query_image_bytes` to base64 once (reused for all pairs)
- Take only the **first 5 results** from `results` (ignore the rest)
- For each of the 5 results, call `judge_pair`
- If score is `None`, skip (do not call feedback)
- Convert score → stars via `score_to_stars`
- Call `POST {gateway_base_url}/feedback` with this payload:
  ```json
  {
    "result_product_id": "<from result payload>",
    "result_image_url": "<from result payload>",
    "result_name": "<from result payload>",
    "store_name": "<from result payload>",
    "category": "<from result payload>",
    "rating": <int 1-5>,
    "source": "auto_judge"
  }
  ```
- Log each judged pair at DEBUG level: `judge: {result_name} → score={score:.2f} stars={stars}`
- Log errors at WARNING level, never raise
- The entire function must be safe to run in a background thread/task

### Implementation — `gateway/main.py`

1. Import `run_judge` from `gateway.judge`
2. Read `GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")` near the other env var reads
3. Read `GATEWAY_BASE_URL = os.getenv("GATEWAY_BASE_URL", "http://localhost:8000")` — needed so the judge can call back to `/feedback`
4. In the `POST /search` handler, **after** the final results list is assembled and **before** returning the response:
   - If `GROQ_API_KEY` is non-empty, add a `BackgroundTasks` parameter to the endpoint and call:
     ```python
     background_tasks.add_task(
         run_judge,
         query_image_bytes,   # the raw bytes of the uploaded/cropped image
         results,             # the final list of result dicts returned to user
         GATEWAY_BASE_URL,
         GROQ_API_KEY,
     )
     ```
   - If `GROQ_API_KEY` is empty, skip silently (judge is opt-in via env var)
5. The search response to the user must not wait for the judge — it returns immediately

### Implementation — `.env.example`

Add these two lines:
```
GROQ_API_KEY=                     # Groq API key — enables live async judge per search
GATEWAY_BASE_URL=http://localhost:8000  # Used by async judge to call back to /feedback
```

### Acceptance criteria
- Search response time is not measurably affected (judge is background)
- With a valid `GROQ_API_KEY`, feedback records appear in `locus_feedback` after a search with `source="auto_judge"`
- Without `GROQ_API_KEY`, no judge runs, no errors
- Only top 5 results are judged, never more
- judge.py has zero imports from gateway/main.py (no circular deps) — it only uses `httpx` or `requests` and stdlib
- All judge errors are logged as warnings and swallowed — they never crash the search endpoint

---

## Change 3 — `is_golden` tag + dev search toggle

### Why
The golden dataset ground-truth items are already indexed in `locus_items` under
`store_name="golden_dataset"`, but they are invisible in normal user searches because
search filters by store/mall context. Adding an `is_golden` payload field lets us
toggle their visibility independently of store filtering, enabling dev-mode validation:
upload a golden query image, enable the toggle, and verify the known ground-truth items
appear in results.

### What to change

#### Part A — Re-index golden items with `is_golden: true`

**File:** `mlops/tools/build_golden_from_lens.py`

When the script upserts points into `locus_items` for the ground-truth result images,
add `"is_golden": True` to the payload dict. This is a non-breaking addition.

Also add a one-off migration function `backfill_is_golden()` at the bottom of the file
(behind `if __name__ == "__main__"` with a `--backfill` CLI flag) that:
- Scrolls all points in `locus_items` where `store_name == "golden_dataset"`
- Sets `is_golden: True` on each via `set_payload`
- Prints progress

#### Part B — `is_golden` as indexed field in Qdrant

**File:** `gateway/main.py`

In the collection setup / `create_payload_index` section (wherever `category_tag`,
`store_name`, etc. are indexed), add:
```python
client.create_payload_index(
    collection_name="locus_items",
    field_name="is_golden",
    field_schema=models.PayloadSchemaType.BOOL,
)
```
Wrap in a try/except that ignores "already exists" errors (same pattern as existing indexes).

#### Part C — Search endpoint dev toggle

**File:** `gateway/main.py`

In the `POST /search` handler, add an optional query parameter:
```python
include_golden: bool = False
```

Default is `False` (production behaviour unchanged).

When `include_golden=True`:
- Remove or relax the `store_name` filter so `store_name="golden_dataset"` items are included
- Add no other changes to the search logic

When `include_golden=False` (default):
- Behaviour is identical to today — golden items are invisible

#### Part D — Visual indicator in results payload

When `include_golden=True`, the search results that come from golden items should have
their existing `store_name="golden_dataset"` visible in the response payload as-is.
No extra field needed — the frontend can use `store_name` to badge them. Do not add
a new field to the response schema for this.

### Acceptance criteria
- `POST /search` with no `include_golden` param: golden items never appear in results
- `POST /search?include_golden=true`: golden items appear alongside real inventory
- Running `python build_golden_from_lens.py --backfill` sets `is_golden=True` on all existing golden points without re-embedding
- New golden items indexed after this change automatically have `is_golden: True`

---

## Change 4 — Replace VSS@5 with Recall@5 and delete the Gemini evaluator

### Why
VSS@5 used Gemini as a proxy for quality because there was no ground truth to measure
against. That problem is now solved: we have a properly annotated golden dataset with
30 queries × 5 ground-truth items each (150 total known-correct items), so we can measure
retrieval quality directly and objectively with no LLM.

The Gemini evaluator (`mlops/evaluate_gemini_judge.py`) is now fully redundant:
- Its evaluation role is replaced by `evaluate_recall.py` (better, faster, free)
- Its per-result scoring role is replaced by the live Groq judge (Change 2)

Delete it. Keeping dead code creates confusion about what the active evaluation
pipeline actually is.

**Recall@K** = of the 150 ground-truth items, how many appear in the top-K results for
their corresponding query.

This is the primary and only quality metric going forward.

### Delete: `mlops/evaluate_gemini_judge.py`

Delete this file entirely. Also delete these Gemini evaluator artifacts if they exist:
- `mlops/vss_cache.json` — Gemini scoring cache, useless without the evaluator
- `mlops/gemini_judge_results.json` — stale output from past Gemini runs

### New file: `mlops/evaluate_recall.py`

Create this file from scratch.

#### What it does
1. Loads `mlops/golden_dataset.json`
2. For each of the 30 query entries:
   - Gets the query image (URL or base64 data URI)
   - Calls `POST /search` on the gateway with that image
   - Collects the returned `product_id` values for top K results
   - Checks how many of the entry's `relevant_product_ids` appear in the top-K set
3. Aggregates across all 30 queries
4. Reports Recall@5, Recall@10, and Recall@25

#### Output format (printed to stdout + optionally logged to MLflow)
```
============================================================
Locus Recall Evaluation
Queries: 30   Ground-truth items: 150
============================================================
Recall@5  :  XX / 150  (XX.X%)
Recall@10 :  XX / 150  (XX.X%)
Recall@25 :  XX / 150  (XX.X%)
------------------------------------------------------------
Per-query breakdown (Recall@5):
  [PASS] query_name — 4/5 hits
  [FAIL] query_name — 1/5 hits
  ...
============================================================
```

#### CLI interface
```
python mlops/evaluate_recall.py [--gateway-url URL] [--k 5,10,25] [--mlflow]
```
- `--gateway-url`: defaults to `http://localhost:8000`
- `--k`: comma-separated K values, defaults to `5,10,25`
- `--mlflow`: if flag present, log results to MLflow under experiment `locus_recall_eval`

#### Implementation notes
- Send the query image to `POST /search` the same way `evaluate_gemini_judge.py` does
  (multipart form-data, or base64 in the request body — match whatever the gateway expects)
- The gateway returns a list of results each with a `product_id` field — use that for matching
- A "hit" = the `product_id` of a ground-truth item appears anywhere in the top-K results
- No LLM calls, no Gemini, no Groq — pure set intersection
- Script should run to completion in under 60 seconds for 30 queries
- If a query fails (network error, gateway down), log a warning and count 0 hits for that query

#### MLflow metrics (when `--mlflow` flag is used)
Log to experiment `locus_recall_eval`:
- `recall_at_5`, `recall_at_10`, `recall_at_25` (float, 0.0–1.0)
- `hits_at_5`, `hits_at_10`, `hits_at_25` (int, out of 150)
- `queries_evaluated` (int)

### Acceptance criteria
- Script runs without errors against a live gateway
- Output shows Recall@5, @10, @25 with per-query breakdown
- No LLM API calls made during execution
- With `--mlflow`, metrics appear in MLflow UI
- Script exits with code 0 on success, code 1 if fewer than 25 queries could be evaluated

---

## Execution Order

Implement in this order — each change is independent but this order avoids re-work:

1. **Change 1** (add `source` to feedback) — foundational, needed by Change 2
2. **Change 2** (Groq judge) — depends on Change 1 being done
3. **Change 3** (golden tag + dev toggle) — independent, can be done anytime
4. **Change 4** (Recall@5 evaluator) — independent, can be done anytime

---

## Do Not Change

- The `locus_feedback` Qdrant collection schema beyond adding the `source` index
- The `locus_items` vector dimension or distance metric
- Any existing API response shapes (all changes are additive)
- Docker Compose service definitions
