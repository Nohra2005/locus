Here's a clean checklist you can save as PLAN.md in your project root. Copy it:


# Locus — Project Completion Checklist

## You (Tatiana)

### Week 1 (now)
- [ ] Continue expanding golden dataset edge cases
- [ ] Fix each bug found → re-run evaluate_recall.py → log to MLflow
- [ ] Target: Recall@5 ≥ 0.96, Precision@5 → 0.80
- [ ] Keep committing to `feature/golden-dataset-accuracy` branch

### Week 2
- [ ] Set up GitHub Actions CI gate (`.github/workflows/ci.yml`)
      → on push to main: run evaluate_recall.py, fail if recall@5 < 0.95
- [ ] Write `docs/tradeoffs.md` — 3 tradeoffs with MLflow evidence:
      1. Fashion-CLIP vs generic CLIP
      2. Recall@K vs Precision@K
      3. LLM judge vs user feedback

### Week 3
- [ ] Final golden dataset sweep
- [ ] Write `tests/test_e2e.py` (calls deployed cloud URL)
- [ ] Open PR, review partner's work, merge everything

---

## Partner

### Week 1 (now)
- [ ] Finish `mlops/export_feedback.py` — export from locus_feedback Qdrant collection
- [ ] Write `mlops/train_projection_head.py` — 2-layer MLP on frozen CLIP, NT-Xent loss
- [ ] Write `mlops/evaluate_and_promote.py` — compare vs baseline, promote if both recall@5 AND judge_score@5 improve
- [ ] Wire into `mlops/retraining_pipeline.py` — full orchestrator (export → train → evaluate → promote → reindex)
- [ ] Write `tests/test_vectorizer.py` — unit tests (no services needed)
- [ ] Write `tests/test_pipeline.py` — integration tests (needs docker compose up)

### Week 2
- [ ] Rewrite `ranking_engine/main.py` — real BLIP caption re-ranker
      → POST /rerank: takes top-25 results + query image, re-scores with BLIP captions
      → combined score: 0.7 * qdrant_score + 0.3 * caption_score
- [ ] Fill in `k8s/deployment.yaml` — gateway, visual-engine, mlops-exporter
- [ ] Fill in `k8s/service.yaml` and `k8s/ingress.yaml`
- [ ] Fix Grafana dashboards — p50/p95 latency, error rate, recall@5, feedback ratio
- [ ] Add rate limiting to gateway (`slowapi`) — 10 req/min on /search

### Week 3
- [ ] Add CLIP confidence histogram to visual_engine (Prometheus, ML-specific signal)
- [ ] Run full retraining pipeline smoke test
- [ ] Confirm K8s applies cleanly

---

## Shared — Git Discipline
- [ ] All work goes on feature branches, never directly to main
- [ ] Every branch gets a PR → other person leaves at least one review comment → merge
- [ ] Branch naming: `feature/what-it-does`

## Rubric Checklist (final check before submission)
- [ ] pytest tests/ → all pass
- [ ] evaluate_recall.py → recall@5 ≥ 0.95 in MLflow
- [ ] Deployed cloud URL responds to /search
- [ ] Grafana shows latency p50/p95, error rate, recall@5, feedback ratio
- [ ] kubectl apply -f k8s/ → no errors
- [ ] docs/tradeoffs.md exists with 3 documented tradeoffs + evidence
- [ ] ranking_engine /rerank endpoint works and is called by gateway
- [ ] Rate limiting triggers on 11th /search per minute
