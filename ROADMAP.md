# Locus — Project Roadmap & Issue Log

> Last updated: April 5, 2026 (Phase 1 completed)
> Current best: **hit@5 = 0.967 | precision@5 = 0.733** (Fashion-CLIP + gateway fallback)

---

## Project Health Summary

The core visual search pipeline is in excellent shape. Fashion-CLIP is already integrated and
performing strongly. The remaining work is about adding intelligence layers (Gemini judge,
fine-tuning), observability (monitoring dashboard), and automation (retraining loop).

---

## Issues Found — Fix Before Proceeding

### 🔴 Critical

**1. Ranking engine is dead code**
The `ranking_engine` service (port 8002) is running in Docker but is **never called** by the
gateway. The gateway sends queries directly to Qdrant, which handles vector similarity search
internally. The ranking engine exists as an idle container burning memory on every `docker compose up`.

→ **Fix**: Either remove it from `docker-compose.yml` now, or repurpose it as the re-ranking
layer when BLIP ensemble is implemented (Phase 3). Don't leave it running unused.

**2. `.env.save` committed to git**
The Qdrant cluster URL was committed in the Apr 3 commit inside `.env.save`. The API key is only
local (not committed), but the cluster URL is now in git history. Not immediately dangerous, but
bad practice.

→ **Fix**: The new `.gitignore` (`.env.*`) prevents any future `.env.*` file from being committed.
If the repo is ever made public, run `git filter-branch` or BFG Repo Cleaner to scrub the history.
Consider rotating the Qdrant API key as a precaution.

---

### 🟡 Important

**3. Gemini judge not pushed**
Two commits reference Gemini judge implementation (Mar 30, Apr 3) but no judge code exists in
the repo. This is the most important missing piece for generating fine-tuning training data and
for automated evaluation.

→ **Fix**: Push existing Gemini judge code. See Phase 1 below.

**4. No tests**
Zero test files anywhere in the project. No unit tests for the vectorizer logic, no integration
tests for the API endpoints, no smoke tests for the pipeline.

→ **Fix**: At minimum, add a `tests/` folder with a `test_smoke.py` that hits each service
health endpoint and a `test_vectorizer.py` that checks category classification on a few known
images. MLflow golden dataset evaluation is a form of integration testing, but unit tests are
still missing.

**5. `repair_db.py` is a footgun**
`tools/repair_db.py` deletes the entire Qdrant `locus_items` collection with no confirmation
prompt. If run by accident it wipes all indexed products.

→ **Fix**: Add a `--confirm` flag and an explicit `input()` prompt before deletion.

**6. `mlflow` in gateway requirements but unused**
`gateway/requirements.txt` lists `mlflow` but gateway code has no MLflow imports or calls.
This adds ~50MB to the Docker image for no reason.

→ **Fix**: Remove `mlflow` from `gateway/requirements.txt`.

---

### 🟢 Low Priority / Nice to Have

**7. No version pinning in most requirements**
`fastapi`, `uvicorn`, `requests`, etc. are unpinned. This can silently break builds when upstream
releases a breaking change.

→ **Fix**: After stabilising dependencies, run `pip freeze` inside each container and lock
versions in requirements files.

**8. CORS is fully open**
`allow_origins=["*"]` in gateway is fine for development but should be restricted to the
frontend origin in production.

**9. Two separate venvs (root + mlops/)**
Root `venv/` was used for `locus_dashboard.py` (now moved to `tools/`). The `mlops/venv/`
is the active evaluation environment. Root venv can be deleted since the dashboard now lives
in `tools/` and should share the mlops venv or have its own requirements file.

---

## Project Structure (Post-Cleanup)

```
locus/
├── gateway/            API hub — search, feedback, scraping, indexing
├── visual_engine/      Fashion-CLIP + YOLO detection + vectorization
├── ranking_engine/     ⚠️ Currently unused — see Issue #1
├── frontend/           React dashboard (Vite + Tailwind + Leaflet)
├── mlops/
│   ├── evaluate_mlflow.py          → run this to benchmark models
│   ├── evaluate_visual_similarity.py
│   ├── diagnose_search.py
│   ├── golden_dataset.json         → ground truth (30 queries, fixed)
│   ├── mlflow.db                   → local MLflow DB (gitignored)
│   └── tools/                      → one-time build/annotation scripts
├── tools/              Root-level utilities (dashboard, repair_db)
├── k8s/                Kubernetes manifests
└── docker-compose.yml
```

---

## Roadmap

### Phase 1 — Baseline & Judge (This Week) ✅ Partially done

**Goal**: Have a clean, reproducible baseline and a working Gemini judge.

- [x] Fashion-CLIP integrated as default model
- [x] MLflow experiments tracking hit@5, precision@5, ndcg@5, mrr
- [x] Golden dataset fixed (30 annotated queries)
- [x] MLflow artifact path fixed (serve-artifacts mode)
- [x] **Run clean baseline evaluation** — use `bash mlops/run_baseline.sh` (script added, run when services are up)
- [ ] **Push Gemini judge code** — wire it into `evaluate_mlflow.py` as `gemini_score@5` metric (partner task)
- [x] Remove dead `ranking_engine` from docker-compose (commented out, reserved for Phase 3)
- [x] Remove `mlflow` from gateway requirements (was unused, added ~50MB to image for nothing)
- [x] Add `--confirm` safety prompt to `tools/repair_db.py`
- [x] Add `tests/test_smoke.py` — service health + category classification + golden dataset validation

**Definition of done**: One clean MLflow run named `fashion_clip_baseline` with all 5 metrics
+ gemini_score logged. This is the number everything else must beat.

---

### Phase 2 — Model Experiments (Next 1–2 Weeks)

**Goal**: Systematically compare model options via MLflow. Each experiment should be a separate
named run so the UI shows a clean comparison table.

**Experiment progression** (run in this order):

1. **`fashion_clip_baseline`** — current system, already at hit@5=0.967 (Phase 1)

2. **`blip_image_encoder`** — swap Fashion-CLIP image encoder for
   `Salesforce/blip-image-captioning-base` vision encoder. Same pipeline, different embeddings.
   Tests whether BLIP's broader pretraining improves fashion retrieval.

3. **`blip_text_channel`** — keep Fashion-CLIP for image embeddings, but at index time
   use BLIP to generate a text caption per product (e.g. "red floral midi dress, short sleeves")
   and store it in the Qdrant payload. At search time, encode the query image caption with
   CLIP's text encoder and add it to the similarity score (weighted: 0.7 image + 0.3 text).
   This is how production visual search engines handle ambiguous queries.
   > Hardware note: BLIP captioning runs **offline at index time only**, not during live search.
   > CPU cost is one-time per product, not per query.

4. **`projection_head_finetuned`** — contrastive fine-tuning using Gemini judge scores +
   user feedback pairs. Do NOT fine-tune full CLIP (too slow on CPU). Instead, train a small
   2-layer MLP projection head on top of frozen Fashion-CLIP embeddings using NT-Xent loss.
   Positive pairs: judge score ≥ 4 OR user rating ≥ 4. Negative pairs: judge score ≤ 2 OR
   user rating ≤ 2. Requires ~200 labeled pairs minimum before training.
   > This is the experiment that requires Gemini judge to be live first.

**How to generate training pairs from Gemini judge**:
The judge runs nightly over the last 24h of search sessions (batch, not live). For each
(query, result) pair in the session log, the judge scores similarity 1–5. These scores plus
existing user feedback in `locus_feedback` form your contrastive training dataset.

---

### Phase 3 — Monitoring Dashboard (Parallel with Phase 2)

**Goal**: A Grafana dashboard the dev team can check to know if the system is healthy.

**Useful metrics to expose** (add to each FastAPI service using `prometheus-fastapi-instrumentator`):

| Panel | Metric | Why it matters |
|-------|--------|---------------|
| Search quality | Rolling avg Gemini score (last 100 searches) | Detects model drift |
| Search quality | Hit@5 from live eval vs golden dataset | Ground truth benchmark |
| User signal | Positive/negative feedback ratio | Real user satisfaction |
| Index health | Items per category in Qdrant | Detects indexing failures |
| Index health | Items in `locus_skipped` (failed) | Spikes = classification breaking |
| System health | Request latency p50/p95 per service | SLA monitoring |
| System health | Error rate per endpoint | Catch regressions early |
| Retraining | Feedback pairs since last retrain | Retraining trigger signal |
| Retraining | Model version currently serving | Know what's deployed |

**Implementation**:
1. Add `prometheus-fastapi-instrumentator` to gateway and visual_engine requirements
2. Add custom `prometheus_client` Gauges for judge scores, feedback counts, collection sizes
3. Add Prometheus + Grafana services to `docker-compose.yml`
4. Import the Grafana dashboard JSON (can be generated from a template)

---

### Phase 4 — Automated Retraining

**Goal**: The system re-trains itself when performance drifts or enough new data accumulates.

**Trigger condition** (choose one or both):
- 200+ new labeled pairs since last retrain (data-driven trigger)
- Gemini score drops below 3.5 average over 7 days (quality-driven trigger)

**Pipeline steps**:
1. Export feedback from `locus_feedback` + judge scores from `locus_judge_scores` (Qdrant)
2. Filter: keep pairs with score ≥ 4 (positive) or ≤ 2 (negative). Discard neutral.
3. Run contrastive fine-tuning (projection head, ~30 min on CPU)
4. Evaluate checkpoint on golden dataset via MLflow
5. **If new model beats current best on both hit@5 AND gemini_score@5 → promote**
6. Re-index entire catalog with new projection head applied to existing embeddings
7. Log model version to Grafana

This can be implemented as a Python script + a Kubernetes CronJob (or simple cron on the host).
The key safety gate is step 5 — the model only goes live if it provably improves things.

---

## Current MLflow Runs (Clean State)

| Experiment | Run Name | hit@5 | precision@5 | ndcg@5 | mrr |
|-----------|----------|-------|-------------|--------|-----|
| locus_visual_similarity | enthused-rook-536 | — | — | — | — |
| locus_search_accuracy | chill-lark-84 (baseline v1) | 0.800 | 0.640 | 0.681 | 0.800 |
| locus_search_accuracy | clip_prompts_fix (intermediate) | 0.733 | 0.593 | 0.624 | 0.717 |
| locus_search_accuracy | clip_prompts_v2_plus_gateway_fallback ⭐ | **0.967** | **0.733** | **0.775** | **0.929** |

> ⭐ Current production model. All future experiments must beat this.

---

## Next Immediate Actions (in order)

1. Run `python3 evaluate_mlflow.py` → name it `fashion_clip_baseline` to establish clean baseline
2. Push Gemini judge code to repo
3. Wire gemini_score into evaluate_mlflow.py
4. Remove ranking_engine from docker-compose (or add TODO comment)
5. Remove `mlflow` from gateway/requirements.txt
6. Start BLIP image encoder experiment (Phase 2, step 2)
