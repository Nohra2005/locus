# Locus — Visual Fashion Search Engine

AI Engineering Capstone — EECE503N / EECE798N
**Team:** Marc El Nawar, Tatiana Nohra

---

## Required API Keys

| Key | Required for | Where to obtain |
|-----|-------------|-----------------|
| `QDRANT_URL` | All services (vector DB) | [cloud.qdrant.io](https://cloud.qdrant.io) — free cluster URL |
| `QDRANT_API_KEY` | All services (vector DB) | Qdrant Cloud dashboard |
| `OPENROUTER_API_KEY` | Gateway (judge), Attribute Tagger (primary) | [openrouter.ai/keys](https://openrouter.ai/keys) |
| `GOOGLE_API_KEY` | Attribute Tagger (fallback only) | Google AI Studio |
| `JWT_SECRET` | Gateway (auth) | Any random string: `python -c "import secrets; print(secrets.token_hex(32))"` |
| `ADMIN_API_KEY` | Gateway (`/admin/*` endpoints) | Any string you choose |

`GOOGLE_API_KEY` is optional — the tagger falls back gracefully to `{}` without it.
`OPENROUTER_API_KEY` is the only key needed to run the full judge + tagger pipeline.

Copy `.env.example` to `.env` and fill in the values before running anything locally.

---

## Live Cloud Deployment

The system is deployed on Azure (Standard_B2s, Ubuntu 22.04) running Kubernetes (K3s).

**Public endpoint:** `http://20.240.203.22:8000`

```bash
# Health check
curl http://20.240.203.22:8000/health

# Visual search (replace path with any clothing image)
curl -X POST http://20.240.203.22:8000/search \
  -F "file=@/path/to/image.jpg" \
  -F "top_k=5"
```

Supporting services on the same VM (accessible for demo/review):

| Service | Port | URL |
|---------|------|-----|
| MLflow | 5000 | `http://20.240.203.22:5000` |
| Grafana | 3000 | `http://20.240.203.22:3000` (admin / admin) |
| Prometheus | 9090 | `http://20.240.203.22:9090` |

---

## Running Locally (Docker Compose)

```bash
git clone https://github.com/tatiananohra/locus.git
cd locus
cp .env.example .env        # fill in QDRANT_URL, QDRANT_API_KEY, OPENROUTER_API_KEY, JWT_SECRET, ADMIN_API_KEY
docker compose up --build -d
curl http://localhost:8000/health
```

Services started: gateway (8000), visual_engine (8001), attribute_tagger (8004),
mlflow (5000), mlops_exporter (8003), prometheus (9090), grafana (3000).

---

## Running on Kubernetes (Azure VM)

```bash
# 1. SSH into the VM
ssh -i locus-vm_key.pem azureuser@20.240.203.22

# 2. Create secrets (one-time)
kubectl create secret generic locus-secrets \
  --from-literal=QDRANT_URL=<value> \
  --from-literal=QDRANT_API_KEY=<value> \
  --from-literal=OPENROUTER_API_KEY=<value> \
  --from-literal=GOOGLE_API_KEY=<value> \
  --from-literal=JWT_SECRET=<value> \
  --from-literal=ADMIN_API_KEY=<value>

# 3. Apply manifests
kubectl apply -f k8s/
kubectl rollout status deployment/gateway

# 4. Verify
curl http://20.240.203.22:8000/health
```

---

## Running the Test Suite

All tests run against a live stack (local or cloud). Start services before running integration/E2E tests.

```bash
# Install test dependencies
pip install -r requirements-test.txt   # or: pip install pytest httpx requests pillow

# Unit tests (no running services needed)
pytest tests/test_vectorizer.py -v

# Integration tests (requires local stack on localhost:8000 and localhost:8001)
pytest tests/test_smoke.py tests/test_pipeline.py tests/test_auth_store.py -v

# End-to-end tests (runs against the live Azure deployment)
pytest tests/test_e2e.py -v
```

CI runs all four suites automatically on every push to `main` via `.github/workflows/ci.yml`.

---

## Running the MLOps Evaluation

### Recall@5 on the golden dataset

```bash
# Against local stack
python mlops/evaluate_recall.py --gateway-url http://localhost:8000 --k 5 --mlflow

# Against cloud deployment
python mlops/evaluate_recall.py --gateway-url http://20.240.203.22:8000 --k 5 --mlflow
```

Results are logged to MLflow experiment `locus_recall_eval`.
Open `http://localhost:5000` (or `http://20.240.203.22:5000`) to browse runs.

### Gemini judge evaluation (model promotion metric)

```bash
python mlops/ci_eval.py
```

Compares current model's average judge score against the baseline in `mlops/ci_baseline.json`.
Exits 0 if the new score meets the promotion threshold.

### Seed historical experiment data (demo only)

```bash
python mlops/seed_mlflow.py         # 12-run hyperparameter ablation study
python mlops/migrate_local_runs.py  # 8 historical runs across 3 experiments
```

---

## Monitoring

**Grafana dashboard:** `http://20.240.203.22:3000` — login: `admin` / `moushou`

The dashboard shows per-service latency (p50, p95), error rates, request throughput,
CLIP confidence histogram, tagger failure rate, and MLflow recall metrics bridged via
the `mlops_exporter` service.

**Prometheus:** `http://20.240.203.22:9090`

Key metrics:

| Metric | Source | What it measures |
|--------|--------|-----------------|
| `locus_searches_total` | gateway | Request volume by category |
| `locus_clip_confidence` | visual_engine | CLIP category confidence histogram (ML signal) |
| `locus_tagger_latency_seconds` | attribute_tagger | VLM call latency |
| `locus_tagger_failures_total` | attribute_tagger | Tagger empty-response rate |
| `http_request_duration_seconds` | all services | p50 / p95 latency per endpoint |

---

## Key Documentation

| File | Contents |
|------|----------|
| `docs/tradeoffs.md` | 5 engineering tradeoffs with evidence (latency, cost, recall, architecture) |
| `PROMPTS.md` | Version history for CLIP category prompts and attribute tagger prompts |
| `mlops/golden_dataset.json` | 35-query ground-truth benchmark |
| `k8s/` | Kubernetes manifests (deployment, service, configmap, secret, ingress) |
| `monitoring/` | Prometheus config, Grafana dashboard JSON |
| `.github/workflows/` | CI pipeline (unit → integration → judge quality gate → E2E), retrain cron |
