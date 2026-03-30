# -*- coding: utf-8 -*-
"""
Visual Similarity Evaluator -- Locus
=====================================

Measures how visually similar your search results are to each query
using the CLIP cosine similarity scores already computed by your system.

NO external API needed -- zero cost, zero rate limits, works immediately.

HOW IT WORKS (plain English)
-------------------------------
When your /search endpoint returns results, each result already contains
a "score" field. That score is the cosine similarity between the CLIP
embedding of the query image and the CLIP embedding of the result image.

  score = 1.0  -> perfectly identical images
  score = 0.8  -> very similar (same color, style, cut)
  score = 0.5  -> loosely related
  score = 0.0  -> completely unrelated

This is already a visual similarity metric. We average the scores of
the top 5 results to get ACS@5 (Average Cosine Similarity at 5).

ACS@5 is your headline metric. When you fine-tune CLIP, this number
will go up -- meaning your results look more similar to the query.

Usage (PowerShell):
  cd "C:\\Users\\User\\Desktop\\Locus Project\\locus\\mlops"
  python -u evaluate_visual_similarity.py

Requirements (already installed):
  python -m pip install requests mlflow
"""

import base64
import io
import json
import os
import sys
import time
import urllib.request
import urllib.error

import requests
import mlflow

# ---- Configuration -----------------------------------------------------------
GATEWAY_URL         = os.getenv("GATEWAY_URL",         "http://localhost:8000")
GOLDEN_DATASET_PATH = os.getenv("GOLDEN_DATASET_PATH", "golden_dataset.json")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME     = "locus_visual_similarity"
TOP_K               = 5
# ------------------------------------------------------------------------------


def p(msg=""):
    """Print with immediate flush so PowerShell shows output right away."""
    print(msg, flush=True)


# ---- Image helpers -----------------------------------------------------------

def get_image_bytes(url: str) -> bytes:
    """
    Get image bytes from a URL or a base64 data URI.
    Your golden_dataset.json stores query images as base64 data URIs.
    """
    if url.startswith("data:"):
        _, b64data = url.split(",", 1)
        return base64.b64decode(b64data)
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (compatible; Locus-Evaluator/1.0)"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read()


# ---- Search helper -----------------------------------------------------------

def run_search(image_bytes: bytes) -> list:
    """POST the query image to /search and return the list of result objects."""
    files    = {"file": ("query.jpg", io.BytesIO(image_bytes), "image/jpeg")}
    response = requests.post(f"{GATEWAY_URL}/search", files=files, timeout=30)
    response.raise_for_status()
    body = response.json()
    if isinstance(body, dict) and "matches" in body:
        return body["matches"]
    if isinstance(body, list):
        return body
    for key in ("results", "items"):
        if isinstance(body, dict) and key in body:
            return body[key]
    return []


def extract_score(result: dict) -> float | None:
    """
    Extract the cosine similarity score from a result object.
    Tries several common field names used by Qdrant / your ranking engine.
    """
    score = (
        result.get("score")
        or result.get("cosine_similarity")
        or result.get("similarity")
        or result.get("payload", {}).get("score")
    )
    if score is not None:
        return float(score)
    return None


# ---- Dataset loading ---------------------------------------------------------

def load_golden_dataset(path: str) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Golden dataset not found at '{path}'.\n"
            "Run  python build_golden_from_lens.py  first."
        )
    with open(path) as f:
        data = json.load(f)
    if not data:
        raise ValueError("Golden dataset is empty.")
    return data


# ---- Main evaluation loop ----------------------------------------------------

def run_evaluation(dataset: list, run_name: str = None) -> dict:

    p("  Connecting to MLflow ...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    p("  MLflow ready.")

    with mlflow.start_run(run_name=run_name) as run:
        p(f"\n  MLflow run ID : {run.info.run_id}")
        p(f"  Experiment    : {EXPERIMENT_NAME}\n")

        mlflow.log_param("num_queries",    len(dataset))
        mlflow.log_param("top_k",          TOP_K)
        mlflow.log_param("gateway_url",    GATEWAY_URL)
        mlflow.log_param("golden_dataset", GOLDEN_DATASET_PATH)
        mlflow.log_param("metric",         "cosine_similarity_clip")
        mlflow.log_artifact(GOLDEN_DATASET_PATH, artifact_path="golden_dataset")

        all_acs5      = []
        all_top1      = []
        per_query_log = []

        for step, entry in enumerate(dataset):
            label   = entry.get("query_name") or entry.get("query_product_id") or f"query_{step}"
            img_url = entry["query_image_url"]
            is_data = img_url.startswith("data:")

            tag = " [embedded]" if is_data else ""
            p(f"\n  [{step + 1:3d}/{len(dataset)}]  {label[:55]}{tag}")

            try:
                query_bytes = get_image_bytes(img_url)

                p("            [search] Running search ...")
                t0         = time.time()
                results    = run_search(query_bytes)
                latency_ms = (time.time() - t0) * 1000
                results_k  = results[:TOP_K]
                p(f"            [ok] Got {len(results_k)} results in {latency_ms:.0f}ms")

                if not results_k:
                    p("            [warn] No results -- skipping")
                    per_query_log.append({"label": label, "error": "no results"})
                    continue

                # Extract cosine similarity scores from results
                scores = []
                for rank, result in enumerate(results_k, start=1):
                    score = extract_score(result)
                    if score is not None:
                        scores.append(score)
                        bar = "#" * int(score * 20) + "." * (20 - int(score * 20))
                        p(f"            [{rank}] similarity={score:.4f}  [{bar}]")
                    else:
                        p(f"            [{rank}] [warn] No score field in result")

                if not scores:
                    p("            [warn] No scores found -- skipping")
                    p("            [info] Check that your /search results include a 'score' field")
                    per_query_log.append({"label": label, "error": "no scores in results", "raw": results_k[:2]})
                    continue

                # ACS@5: Average Cosine Similarity at 5
                acs5   = sum(scores) / len(scores)
                top1   = scores[0]  # score of the best result

                all_acs5.append(acs5)
                all_top1.append(top1)

                p(f"\n            [stats] ACS@5 = {acs5:.4f}   Top-1 score = {top1:.4f}")

                mlflow.log_metrics({
                    "q/acs5":       acs5,
                    "q/top1_score": top1,
                    "q/latency_ms": latency_ms,
                }, step=step)

                per_query_log.append({
                    "label":      label,
                    "acs5":       acs5,
                    "top1_score": top1,
                    "scores":     scores,
                    "latency_ms": latency_ms,
                })

            except urllib.error.URLError as e:
                p(f"            [error] Image fetch failed: {e}")
                per_query_log.append({"label": label, "error": str(e)})
            except requests.HTTPError as e:
                p(f"            [error] Search request failed: {e}")
                per_query_log.append({"label": label, "error": str(e)})
            except Exception as e:
                p(f"            [error] {e}")
                per_query_log.append({"label": label, "error": str(e)})

        # ---- Aggregate -------------------------------------------------------
        n_ok = len(all_acs5)
        if n_ok == 0:
            p("\n  [warn] No queries could be evaluated.")
            p("  Make sure docker compose is running (docker compose up -d)")
            return {}

        headline = {
            "avg_acs5":       sum(all_acs5) / n_ok,
            "avg_top1_score": sum(all_top1) / n_ok,
            "n_evaluated":    n_ok,
            "n_failed":       len(dataset) - n_ok,
        }

        mlflow.log_metrics(headline)

        results_path = "visual_similarity_results.json"
        with open(results_path, "w") as f:
            json.dump(per_query_log, f, indent=2)
        mlflow.log_artifact(results_path)

        def bar(v): return "#" * int(v * 20) + "." * (20 - int(v * 20))

        acs5_val = headline["avg_acs5"]
        top1_val = headline["avg_top1_score"]

        p()
        p("  " + "=" * 58)
        p("  VISUAL SIMILARITY EVALUATION SUMMARY")
        p("  " + "=" * 58)
        p(f"  Queries evaluated  : {n_ok} / {len(dataset)}")
        p()
        p(f"  ACS@5 (headline)   : {acs5_val:.4f}  [{bar(acs5_val)}]")
        p(f"  Avg top-1 score    : {top1_val:.4f}  [{bar(top1_val)}]")
        p()
        p("  What ACS@5 means:")
        p("    >= 0.85  Excellent -- results are near-identical to query")
        p("    >= 0.70  Good      -- strong visual similarity")
        p("    >= 0.55  Fair      -- loosely related results")
        p("     < 0.55  Poor      -- results don't match query visually")
        p()
        p("  This score will go UP when you fine-tune your CLIP model.")
        p("  Run this script before and after each fine-tuning round")
        p("  to measure improvement.")
        p("  " + "=" * 58)

        return headline


# ---- Entry point -------------------------------------------------------------

def main():
    p()
    p("=" * 64)
    p("  Locus Visual Similarity Evaluator")
    p("  Uses CLIP cosine similarity -- no external API needed")
    p("=" * 64)
    p(f"  Dataset        : {GOLDEN_DATASET_PATH}")
    p(f"  Gateway        : {GATEWAY_URL}")
    p(f"  MLflow storage : {MLFLOW_TRACKING_URI}")
    p(f"  Top-K          : {TOP_K}")
    p()

    dataset = load_golden_dataset(GOLDEN_DATASET_PATH)
    n_data  = sum(1 for e in dataset if e.get("query_image_url", "").startswith("data:"))
    p(f"  Loaded {len(dataset)} queries  ({n_data} embedded images, {len(dataset)-n_data} URLs)")
    p()

    p("  Make sure Docker is running before continuing:")
    p("    docker compose up -d   (run from the locus root folder)")
    p()

    run_name = input("  Run name (press Enter for auto-timestamp): ").strip() or None
    confirm  = input(f"  Start evaluation? [y/N]: ").strip().lower()
    if confirm != "y":
        p("  Aborted.")
        return

    headline = run_evaluation(dataset, run_name)

    if headline:
        p()
        p("  [ok] Done! Results saved.")
        p("  To view in MLflow:")
        p("    mlflow ui --backend-store-uri sqlite:///mlflow.db")
        p("    Then open: http://localhost:5000")
    p("=" * 64)
    p()


if __name__ == "__main__":
    main()
