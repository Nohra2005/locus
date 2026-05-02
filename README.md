# Project Scope Statement: Locus Visual Search Engine

## 1. Problem Statement (The "Why")
**Current Issue:**
Online shopping has become a major part of today’s fashion industry. The main inconvenience is clients buy without trying/seeing the product and have to wait for shipping. While many shoppers really enjoy going out to the mall, physical shopping is unarguably draining, inefficient and often unfruitful: shoppers can spend countless hours looking for a specific product they have in mind or they have found on Pinterest/Instagram but have no idea where to find it. Some countries like Lebanon lack a centralized marketplace where people can easily order from a wide variety of products. This emphasizes the need for Lebanese shoppers to physically go out to malls.

**Impact:**
Locus merges best of both worlds: convenience and precision of online shopping with the product experience of physical shopping.

## 2. Project Objective (The "What")
To design and develop a clothing recommendation system that returns similar items ranked by similarity and nearness that allows users to upload an image and retrieve visually similar inventory items with a focus on accuracy and speed.

## 3. In-Scope (Features & Functionalities)

### Core Functionality (The "What")
* **Visual Search Pipeline:** End-to-end processing of user uploads, including automatic background removal, object isolation (ROI Normalization), and vectorization.
* **Multi-View Product Indexing:** Indexes inventory items as multi-vector "folders" (Front, Back, Lifestyle) to ensure accurate retrieval regardless of the input angle.
* **Inventory Localization:** Maps visual search results to specific physical store locations to solve the "shipping delay" pain point.
* **Smart Categorization:** Automatically identifies clothing categories (e.g., "Dress", "Coat") to filter search results.
* **User Authentication:** To keep track of search/purchase history for recommendation engines.

### Algorithmic Capabilities (The "How")
* **Hybrid Retrieval Engine:** Combines Vector Similarity (Visual Match) with Hard Filters (Category, Location, Price) for high-precision results.
* **Adversarial Input Robustness:**
    * Rejects empty/ghost images where background removal failed.
    * Filters out predictions with <45% confidence.

### Recommendation & Personalization Engines
* **"User Persona" History Recommendations:**
    * **Concept:** Builds a dynamic "Taste Profile" for each user based on their interaction history.
    * **Mechanism:** The system cycles through items in a user's history to generate recommendations (Stochastic Sampling).
* **"Vibe-Check" Outfit Completion:**
    * **Concept:** Suggests complementary items (e.g., accessories) that match the aesthetic of the current search.
    * **Mechanism:** Uses Zero-Shot Style Anchoring (via CLIP) to classify the search item's style (e.g., "Bohemian", "Minimalist"). The system then queries the inventory for complementary categories (e.g., Shoes) that share that specific style tag, ensuring a coherent outfit suggestion.

### User Interface
* **Visual Dashboard:** A responsive web interface featuring:
    * **Smart Crop Tool:** Allows users to manually adjust the focus area.
    * **AI Vision Debugger:** Transparent view of the background-removed input.
    * **Local Availability Map:** Displays store locations for matched items.
    * **Recommendation Engine:** Recommends based on search history.
* **Dashboard for Shop Owners:** To upload their inventory.

## 4. Out-of-Scope (The "No-Go" Zone)
* Multi-object detection (detecting a hat and shoes simultaneously).
* Mobile app development (Web only).
* Integration with live payment gateways.
* A system that can 100% detect the category of random inserted objects and affirm with certainty that it is not a piece of clothing (not crucial).

## 5. Technical Constraints & Requirements
* **Performance:** Search latency can be compromised but to a certain extent. The model should be able to find similar looking items and especially “exact match”. We need more accuracy than latency.
* **Infrastructure:** Gateway is for uploading the image, visual engines prepare the image and detect the category, and the ranking search through the quadrant database for the best match.
* **Data:** Retailers catalogue.
* **Hardware:** Must run on standard CPU architecture (no GPU requirement).

## 6. Success Metrics (KPIs)
* **Accuracy:** System correctly categorize items in precise groups and return accurate similar items. At least 8 pictures over 10 are similar to what I am looking for.
* **Robustness:** System successfully rejects low confidence predictions (with 45% resemblance and less).
* **Speed:** End-to-end processing must take maximum 15s.

## 7. System Architecture

```
User / Browser
      │ HTTPS
      ▼
┌─────────────────────────────────┐
│  Gateway (FastAPI, port 8000)   │  ← EEP: orchestration, auth, rate-limiting
│  JWT auth · slowapi · Prometheus│
└────────┬──────────────┬─────────┘
         │              │
    /embed, /detect  /classify, /refine
         │              │
         ▼              ▼
┌─────────────┐  ┌──────────────────┐
│visual_engine│  │ attribute_tagger │  ← IEPs
│ (CLIP+YOLO) │  │ (Gemini 2.0 Flash│
│  port 8001  │  │   port 8004)     │
└──────┬──────┘  └──────────────────┘
       │ vectors
       ▼
  Qdrant Cloud (vector DB)

Background services (same VM):
  MLflow :5000  ← experiment tracking
  mlops_exporter :8003 ← ML metrics → Prometheus :9090 → Grafana :3000
  link_monitor  ← catalog link health (async)

CI/CD (GitHub Actions):
  unit-tests → integration-tests → judge quality gate → E2E tests
  retrain.yml (cron every 2 days) → LoRA fine-tune → promote → SSH deploy → hot-reload
```

## 8. Secrets Management

Locus uses three API keys: `QDRANT_API_KEY`, `OPENROUTER_API_KEY`, and `GOOGLE_API_KEY`. Each is never committed to the repository. The table below shows how they are injected per environment.

| Environment | Mechanism | Where defined |
|---|---|---|
| Local dev | `.env` file (git-ignored) | Copy `.env.example`, fill values |
| Docker Compose | `env_file: .env` in each service | Same `.env` file, loaded at runtime |
| GitHub Actions CI | GitHub repository Secrets | Settings → Secrets → Actions |
| Kubernetes (Azure VM) | `locus-secrets` Secret object | `kubectl create secret generic locus-secrets --from-env-file=.env` |

**Kubernetes secret creation (one-time, run on the VM):**
```bash
kubectl create secret generic locus-secrets \
  --from-literal=QDRANT_API_KEY=<value> \
  --from-literal=OPENROUTER_API_KEY=<value> \
  --from-literal=GOOGLE_API_KEY=<value>
```

Pods read secrets via `secretKeyRef` in `k8s/deployment.yaml` — keys are injected as environment variables and never written to disk or logs.

**Key rotation:** rotate a key by updating the GitHub Secret and re-running the retrain/deploy workflow, or by patching the K8s secret with `kubectl create secret generic locus-secrets --from-literal=KEY=<new> --dry-run=client -o yaml | kubectl apply -f -` and restarting the affected deployment.

## 9. Cloud Deployment & Cost Estimate

**Deployment:** Azure VM `Standard_B2s` (2 vCPU, 4GB RAM) running Docker Compose with 7 services. Public IP: `20.240.203.22`.

| Component | Cost |
|---|---|
| Azure Standard_B2s VM | ~$35/month |
| Qdrant Cloud (free tier, 1GB) | $0 |
| OpenRouter (Gemini judge, ~500 calls/day) | ~$2–5/month |
| Google Gemini API (attribute tagger fallback) | ~$1–3/month |
| GitHub Actions (CI, self-hosted runner on VM) | $0 |
| **Total estimated** | **~$38–43/month** |

**How to start the system:**
```bash
ssh -i locus-vm_key.pem azureuser@20.240.203.22 "cd ~/locus && docker compose up -d"
```

## 10. Assumptions & Risks
* **Assumption:** User photos will have reasonable lighting and resolution.
* **Risk 1: Background Removal Failure.**
    * **Issue:** `rembg` might fail on white-on-white images or complex textures.
    * **Mitigation:** Implemented alpha-channel check to detect "ghost" images and reject them early.
* **Risk 2: Inventory Desynchronization (Ghost Stock).**
    * **Issue:** Physical sales in stores may not be immediately reflected in the digital database, leading users to drive to a store for an item that was just sold.
    * **Mitigation:**
        * **Timestamp Transparency:** Display "Last Updated: [Date]" prominently on every item card.
        * **"Call to Reserve" CTA:** A primary button connecting the user directly to the shop's Phone/WhatsApp to confirm availability before driving.
        * **Weekly Refresh Model:** Shop owners are required to upload a fresh Excel/CSV export weekly rather than real-time syncing.
* **Risk 3: Model Misclassification.**
    * **Issue:** The AI might wrongly classify a specific item (e.g., labeling a "Skirt" as a "Dress").
    * **Mitigation (Feedback Loop):** A "Report/Correct" feature allows users to flag incorrect categories. These corrections are stored and used to fine-tune the model or exclude specific items from future search results.
