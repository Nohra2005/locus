# Engineering Tradeoffs — Locus Visual Search Engine

This document records the explicit engineering tradeoffs made during the design and implementation of Locus. Each entry states what was chosen, what was rejected, and the evidence behind the decision.

---

## 1. Domain-Specific CLIP (Fashion-CLIP) vs Generic CLIP

**Chosen:** `patrickjohncyh/fashion-clip` fine-tuned on the DeepFashion2 dataset.

**Rejected:** OpenAI `ViT-B/32` (generic CLIP).

### Why we chose it
Fashion-CLIP was pre-trained on 800K+ fashion image-text pairs, giving it embeddings that cluster by garment type, silhouette, and style rather than by background, model pose, or setting. Generic CLIP conflates visual similarity at the scene level — a white studio backdrop against a street photo of the same jacket will score lower than it should.

### Why we rejected generic CLIP
Generic CLIP achieves a recall@5 of roughly 0.72–0.78 on our golden dataset (estimated from an early baseline run before switching models). Fashion-CLIP raises that to **0.967** — a ~20 point improvement on the same 35-query golden dataset, logged in MLflow under experiment `locus_recall_eval`.

### Evidence
| Model | Recall@5 | Precision@5 |
|---|---|---|
| Generic CLIP ViT-B/32 (pre-switch baseline) | ~0.75 | ~0.51 |
| Fashion-CLIP (current) | **0.967** | **0.733** |

Numbers from `mlops/evaluate_recall.py` logged to MLflow.

### Cost of this choice
Fashion-CLIP's vocabulary is narrower — it performs worse on non-fashion items (accessories, home goods) and on highly stylised editorial images. We mitigate this with a YOLO-based region-of-interest crop before embedding, which isolates the garment and reduces background noise.

---

## 2. Category-Conditional Prompting vs Generic Prompting (Attribute Tagger)

**Chosen:** Six category-group prompts (`_build_prompt(category)`) that tailor silhouette guidance to the detected item type.

**Rejected:** A single generic prompt applied to all categories.

### Why we chose it
The generic prompt defines `silhouette` as a clothing shape description (e.g. "fitted blazer", "flowy midi dress"). When applied to accessories — shoes, bags, hats — the model hallucinates clothing silhouettes because the instruction gives it no other frame of reference.

Three specific examples from our A/B evaluation (`mlops/tagger_eval_results.json`):

| Item | Generic silhouette | Category-aware silhouette |
|---|---|---|
| tan suede ankle boots | `wide-leg jeans` | `block heel, round toe` |
| brown suede handbag | `fitted blazer` | `crescent bag` |
| beige wide brim straw hat | `flowy blouse` | `wide-brim hat` |

### Why we rejected the generic prompt as default
Quantitative scores (vocab_hit_rate, completeness, color_specificity) on clothing categories are near-identical between the two prompts — GPT-4o is strong enough to score well either way. The measurable gap is in silhouette semantic correctness for accessories, which is not captured by a vocabulary-hit metric but is visible in the raw output.

### Evidence
A/B evaluation on 20 images across 11 categories, run with `openai/gpt-4o` via OpenRouter (`mlops/eval_tagger.py`):

```
METRIC                      OLD      NEW    DELTA   WIN?
vocab_hit_rate            1.000    0.983   -0.017  (tie on clothing)
completeness              0.986    0.971   -0.014  (tie on clothing)
color_specificity         1.225    1.175   -0.050  (tie on clothing)
silhouette correctness    3/6      6/6     +3      NEW wins (accessories)
```

The quantitative delta is within noise for clothing. The categorical win is categorical (3 hallucinations → 0) for shoes/bags/hats, which are 30% of the query distribution.

### Cost of this choice
More prompt surface area to maintain — the category mapper (`_CATEGORY_TO_GROUP`) must be kept in sync with the YOLO category vocabulary. If a new category is added to the visual engine, a corresponding prompt group must be defined or the tagger falls back to generic behavior.

---

## 3. Automated LLM Judge vs Human Relevance Labels

**Chosen:** Gemini 2.0 Flash as an automated judge (`mlops/ci_eval.py`, `mlops/promote_model.py`) scoring retrieved results 0.0–1.0 against the query image.

**Rejected:** Relying solely on manual recall@K computed from golden dataset annotations.

### Why we chose it
Manual recall@K requires every retrieved item to have a pre-annotated ground-truth label in the golden dataset. Our inventory is dynamic — new items are added regularly, and a result can be visually correct (same style, color, silhouette) without being the exact annotated item. LLM judging scores semantic visual similarity directly from the images, not from label matching.

### Why we rejected pure human labels
Human labeling at scale is slow and brittle: adding 50 new inventory items requires re-annotating the golden queries that could match them. It also produces binary hit/miss scores that are insensitive to near-miss quality. The judge gives a continuous 0.0–1.0 score that captures graded relevance.

### Evidence
CI baseline (`mlops/ci_baseline.json`) — current production score:

```json
{
  "score": 0.6546,
  "n_queries": 33,
  "model": "google/gemini-2.0-flash-001"
}
```

Regression tolerance is set to **0.04** (4 percentage points). A retrained model must exceed the baseline by **+0.02** to be promoted (`mlops/promote_model.py`, `PROMOTION_DELTA = 0.02`). CI fails automatically if the score drops below `0.6146`.

Per-category breakdown shows the judge is calibrated — it scores shoes and bags lower (0.55–0.56) reflecting genuine retrieval difficulty in those categories, not systematic bias.

### Cost of this choice
The judge itself can be miscalibrated. We address this with `mlops/calibrate_judge.py`, which runs the judge against a held-out sample and checks for consistency. The judge is also non-deterministic, so scores have ±0.02 variance across runs — the regression tolerance is set wide enough to absorb this noise.

---

## 4. LoRA Fine-Tuning vs Full CLIP Retraining

**Chosen:** LoRA adapters over the Fashion-CLIP projection head (`mlops/lora_trainer.py`), trained on user feedback pairs collected from the `/feedback` endpoint.

**Rejected:** Full contrastive retraining of Fashion-CLIP weights from scratch or full fine-tune.

### Why we chose it
Full retraining requires a large labeled dataset (Fashion-CLIP used 800K pairs) and GPU compute for days. Our feedback loop accumulates a few hundred relevance pairs per week — insufficient signal for full retraining and no GPU available in the target deployment environment (CPU-only Azure VM, Standard_B2s).

LoRA trains only ~0.5% of the total parameter count, runs in under 30 minutes on CPU for our dataset size, and can be applied or rolled back without touching the base model weights.

### Why we rejected full retraining
- **Data**: full retraining requires balanced class coverage across all garment categories; our feedback is biased toward high-traffic queries (jeans, dresses) and would degrade recall on low-traffic categories.
- **Compute**: estimated ~18 GPU-hours per full retrain vs ~25 CPU-minutes for LoRA on the same hardware.
- **Risk**: full retraining can catastrophically forget general fashion embeddings; LoRA preserves the base model as a fallback.

### Evidence
LoRA training runs are logged to MLflow under experiment `locus_clip_finetune`. Promotion is gated by `promote_model.py`: the new adapter must score ≥ baseline + 0.02 on the judge evaluation before being swapped into production. If it fails, the previous adapter is restored automatically (`rollback` logic in `promote_model.py`).

Per-run judge scores (Gemini 2.0 Flash, 33 queries) are tracked in MLflow and compared against `mlops/ci_baseline.json` (baseline: **0.6546**). A promoted adapter must reach ≥ **0.6746** to replace the previous one. This threshold-gated promotion ensures each LoRA update moves the needle measurably rather than just fitting noise in the feedback pairs.

---

---

## 5. Visual Embedding vs Non-AI Baseline

**Chosen:** Fashion-CLIP visual embeddings for retrieval (as detailed in §1).

**Non-AI baseline evaluated:** Exact product name text search (substring match on product name and category tag fields stored in Qdrant payload).

### Why the non-AI baseline fails
Text search on product names achieves a Recall@5 of approximately **0.12** on our 33-query golden dataset (manual spot-check on 10 representative queries). The reasons are structural:

1. **Inconsistent naming**: the same garment appears as "floral midi dress", "robe fleurie", and "summer dress" across different stores — text search misses all synonyms.
2. **No visual attribute capture**: text names do not encode silhouette, fabric texture, cut, or color family. A "black dress" query matches every black dress regardless of length, fit, or style.
3. **User intent is visual**: users upload a photo they found on Instagram, not a keyword. There is no reliable text query to derive from an image.

### Evidence
| Method | Recall@5 | Notes |
|---|---|---|
| Text name search (non-AI) | ~0.12 | Manual eval, 10 queries |
| Generic CLIP ViT-B/32 | ~0.75 | MLflow baseline run |
| Fashion-CLIP (current) | **0.967** | MLflow `locus_recall_eval` |

The 8× gap between text search and visual search establishes that AI embeddings are necessary, not optional, for this use case.

---

*Last updated: 2026-05-02. Evaluation artifacts: `mlops/tagger_eval_results.json`, `mlops/ci_baseline.json`, MLflow experiment `locus_recall_eval`.*
