# Locus — Project Completion Checklist
> Last updated: April 21, 2026

---

## 🔴 Fix First (Blocking)

~~- [ ] **Fix CLIP regression**~~ ✅ Done
~~- [ ] **Wire new API key**~~ ✅ Done

---

## ✅ Already Done

- [x] LLM judge — OpenRouter + Gemini 2.0 Flash, rate-limited, background task, stores feedback; fallback judge added
- [x] Judge calibration — `mlops/calibrate_judge.py` runs against golden dataset, outputs pass/warn/fail
- [x] Retraining pipeline — `build_training_pairs.py` → `lora_trainer.py` → `promote_model.py` → rollback logic
- [x] LoRA fine-tuning — `mlops/Dockerfile.retrain`, runs on 48h schedule via `mlops_retrain` service
- [x] MLflow experiment tracking — logs recall@K, promotion decisions, run names
- [x] Grafana dashboards — 15+ real panels (collection sizes, recall@5, judge calibration, link health)
- [x] Prometheus metrics — scrapes gateway, visual_engine, mlops_exporter
- [x] Link health monitor — checks broken image URLs every 5 days
- [x] Golden dataset evaluation — `evaluate_recall.py`, recall@K, MLflow logging
- [x] Docker Compose — gateway, visual_engine, mlflow, mlops_retrain, mlops_exporter, prometheus, grafana, **attribute_tagger**
- [x] Frontend — search history, product detail sheet, saved items
- [x] Cloud deployment — publicly accessible URL
- [x] **Attribute tagger** (`attribute_tagger/`, port 8004) — Gemini-powered fashion attribute extraction; Prometheus Counter + Histogram; wired into docker-compose; satisfies §4.2 second real IEP requirement
- [x] Corrupt image auto-detection — gateway skips broken images before indexing
- [x] System hardening — non-fashion detection, fallback judge, AI rerank in gateway

---

## ❌ Still Missing (Rubric Requirements)

### Tests — §8 "Non-Negotiable"
- [x] `tests/test_smoke.py` — 16 smoke tests (stale ranking_engine ref removed)
- [x] `tests/test_vectorizer.py` — unit tests: shoe sub-type (16 cases), alias mapping (10 cases), canonical labels; no services needed
- [x] `tests/test_pipeline.py` — integration: search schema, category detection, detect, feedback round-trip, rate limiting, CORS, attribute tagger, corrupt image, non-fashion handling
- [x] `tests/test_e2e.py` — hits `$GATEWAY_URL`, validates schema, non-empty results, image URLs, feedback; skipped automatically if URL not set

### CI/CD — §6 Git Discipline
- [x] `.github/workflows/ci.yml` — unit tests (job 1, no services) → integration + judge quality gate (job 2, full stack); fails if judge avg < 0.65
- [x] `mlops/ci_eval.py` — judge-based quality gate replacing recall@K; 5 diverse golden queries, Gemini scores top-3 results

### Kubernetes — §9 Required
- [x] `k8s/deployment.yaml` — gateway, visual-engine, attribute-tagger, mlops-exporter; readiness probes; secret refs
- [x] `k8s/service.yaml` — LoadBalancer for gateway (port 80→8000), ClusterIP for visual-engine, attribute-tagger, mlops-exporter
- [x] `k8s/ingress.yaml` — nginx, TLS, 10MB proxy-body-size annotation
- [x] `k8s/configmap.yaml` — QDRANT_URL, GATEWAY_BASE_URL, VISUAL_HOST, TAGGER_HOST; **fill in real values before `kubectl apply`**

### Second Real IEP — §4.2 ✅ Fulfilled by attribute_tagger
- [x] `attribute_tagger/` — live service, real model calls (Gemini), exposes POST endpoint, Prometheus metrics, wired in docker-compose

### Security & Robustness — §12
- [x] `slowapi` rate limiting — 10/min per IP on `/search`, 5/min on `/detect`
- [x] CORS — restricted via `CORS_ORIGINS` env var (defaults to `*` in dev; set to your domain in `.env` on the VM)
- [x] Add 10MB max payload size on image upload endpoints

### Tradeoffs Documentation — §5 Required
- [ ] Create `docs/tradeoffs.md` with 3 tradeoffs + evidence (MLflow numbers):
  1. Fashion-CLIP vs generic CLIP — domain accuracy vs generality
  2. LoRA fine-tuning vs full CLIP retrain — cost/speed vs ceiling
  3. LLM judge (automated) vs user feedback (real) — coverage vs ground-truth quality

### ML-Specific Monitoring Signal — §11
- [~] Detection histogram exists in `visual_engine/main.py:19` (`locus_detections_histogram`), and `attribute_tagger` has its own Histogram — but rubric asks for a **CLIP confidence score** histogram on `/vectorize` specifically; `category_confidence` is returned but not tracked as a Prometheus metric yet
- [ ] Add `locus_clip_confidence` Histogram in `visual_engine/main.py` — observe `confidence` on each `/vectorize` call

---

## Work Division

| Task | Owner | Status |
|------|-------|--------|
| Fix CLIP regression | **You** | ✅ Done |
| Wire new API key | **You** | ✅ Done |
| Attribute tagger (IEP #2) | **You** | ✅ Done |
| K8s manifests (all 4 files) | **You** | ✅ Done |
| Expand golden dataset, re-run eval | **You** | ❌ Not started |
| CI/CD GitHub Actions | **You** | ✅ Done |
| All tests (unit, integration, e2e) | **You** | ✅ Done |
| Tradeoffs doc | **You** | ❌ Not started |
| Unit + integration tests | **Partner** | ❌ Not started |
| Rate limiting + CORS lock-down | **You** | ✅ Done |
| CLIP confidence histogram | **Partner** | 🟡 Partial |

---

## Final Rubric Check (before submission)

- [ ] `pytest tests/ -v` → all pass (smoke tests exist; unit + integration + e2e still needed)
- [ ] `evaluate_recall.py` → recall@5 ≥ 0.95 logged in MLflow
- [ ] Deployed cloud URL responds to `/search` with results
- [ ] Grafana shows p50/p95 latency, error rate, recall@5, feedback ratio, CLIP confidence histogram
- [ ] `kubectl apply -f k8s/` → no errors (update configmap values + create `locus-secrets` Secret first)
- [ ] `docs/tradeoffs.md` → 3 tradeoffs with MLflow evidence
- [ ] `attribute_tagger` POST endpoint callable and wired into at least one gateway flow
- [ ] Rate limit triggers on 11th `/search` per minute
- [ ] CORS restricted to deployed frontend domain
- [ ] GitHub PR history shows feature branches + review comments
