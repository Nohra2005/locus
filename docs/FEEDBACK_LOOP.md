# Locus Closed-Loop Retraining Pipeline

This document describes the automated feedback → training → deployment cycle that continuously improves retrieval quality without human intervention.

---

## Overview

Locus implements a full closed-loop ML pipeline: user star ratings are converted into contrastive training pairs, used to fine-tune a LoRA adapter on the running Fashion-CLIP model, evaluated against a quality gate, and hot-swapped into production — all without restarting any container or requiring manual action.

```
User submits star rating
        │
        ▼
POST /feedback  (gateway/main.py)
  • Converts rating → training_signal (positive / neutral / negative)
  • Stores in Qdrant `locus_feedback` collection with product embedding ref
        │
        ▼
Trigger condition met  (mlops/retrain_clip.py)
  • Condition A: feedback count ≥ FEEDBACK_RATE × catalog size
  • Condition B: ACS@5 score drops below MIN_ACS_THRESHOLD
  • Trigger: GitHub Actions cron (every 48h) OR POST /trigger-retrain
        │
        ▼
Build training pairs  (mlops/build_training_pairs.py)
  • Fetches positive/negative feedback from Qdrant
  • Constructs (anchor, positive) image pairs with augmentation
  • Writes pairs_cache/pairs_manifest.json
        │
        ▼
LoRA fine-tuning  (mlops/lora_trainer.py)
  • Loads Fashion-CLIP (patrickjohncyh/fashion-clip)
  • Injects LoRA adapters (r=4) into q_proj / v_proj of vision transformer
  • Trains with InfoNCE loss, temperature τ=0.07
  • 15% held-out validation split → logs val_loss + val_recall@5 to MLflow
  • Saves adapter to lora_runs/<run_id>/lora_adapter/
        │
        ▼
Quality gate  (mlops/promote_model.py)
  • Runs Gemini 2.0 Flash judge on 33 golden queries
  • Compares new score against ci_baseline.json (baseline: 0.6546)
  • PROMOTE if new_score ≥ baseline + PROMOTION_DELTA (0.02)
  • ROLLBACK if new_score < baseline (restores previous adapter)
        │
        ▼  (on promotion)
Hot-swap adapter  (visual_engine POST /reload-adapter)
  • Loads new adapter weights without restarting the container
  • Atomically replaces the vision encoder in-place (~30s)
  • No downtime for in-flight search requests
        │
        ▼
Update baseline  (mlops/ci_eval.py --set-baseline)
  • Writes promoted score to ci_baseline.json
  • Future promotions must beat this new bar
```

---

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `FEEDBACK_RATE` | `0.10` | Fraction of catalog size that triggers a retrain when feedback accumulates past it |
| `MIN_ACS_THRESHOLD` | `0.70` | Quality drift threshold — retrain triggered if ACS@5 drops below this |
| `PROMOTION_DELTA` | `0.02` | New adapter must beat baseline judge score by this margin to be promoted |
| `MAX_TRAIN_STEPS` | `300` | LoRA training steps (300 steps ≈ 25 CPU-minutes on Standard_B2s) |
| `VAL_SPLIT` | `0.15` | Fraction of pairs held out for validation during training |
| `BATCH_SIZE` | `4` | Training batch size (effective batch = 4 × GRAD_ACCUM_STEPS = 16) |

---

## Failure Behavior

| Failure point | What happens |
|---|---|
| Not enough training pairs | `retrain_clip.py` skips the run and logs a warning to MLflow |
| Training loss explodes / NaN | Trainer catches exception per step, skips bad batch, continues |
| Judge score below baseline | `promote_model.py` restores the previous adapter; baseline unchanged |
| `/reload-adapter` fails on visual_engine | Error logged; container keeps running with previous adapter; retrain pipeline returns failure status |
| Judge API unavailable | Retrain aborts; no promotion attempted; previous adapter preserved |

---

## MLflow Tracking

All retrain runs are logged under experiment `locus_clip_finetune`:

| Metric | Description |
|---|---|
| `train_loss` | InfoNCE loss on training batch (logged every `LOG_EVERY_STEPS`) |
| `val_loss` | InfoNCE loss on held-out 15% validation split |
| `val_recall_at_5` | Fraction of val anchors whose positive appears in top-5 (by cosine similarity) |
| `final_val_loss` | Val loss at end of training |
| `final_val_recall_at_5` | Val recall@5 at end of training |
| `judge_score_new` | Gemini judge score for the candidate adapter |
| `judge_score_old` | Previous baseline score |
| `promoted` | Boolean — whether the adapter was promoted to production |

Browse runs at `http://<VM_IP>:5000` (MLflow UI, port 5000).

---

## Files

| File | Role |
|---|---|
| `gateway/main.py` — `POST /feedback` | Collects user ratings, stores in Qdrant |
| `gateway/main.py` — `POST /trigger-retrain` | Manual trigger endpoint (admin-only) |
| `mlops/retrain_clip.py` | Orchestrator: checks conditions, runs train → promote cycle |
| `mlops/build_training_pairs.py` | Builds (anchor, positive) pairs from feedback |
| `mlops/lora_trainer.py` | LoRA fine-tuning with val split and MLflow logging |
| `mlops/promote_model.py` | Quality gate: judge evaluation + promote/rollback logic |
| `mlops/ci_baseline.json` | Current production quality baseline (score + model + per-category) |
| `visual_engine/main.py` — `POST /reload-adapter` | Hot-swaps the adapter in-place |
| `.github/workflows/retrain.yml` | Cron trigger (every 48h) on the self-hosted runner |
