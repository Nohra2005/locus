"""
MLFlow Evaluation Script — Locus Visual Search
===============================================
Loads your golden dataset, runs every query image through the Locus /search
endpoint, computes retrieval metrics, and logs everything to MLFlow.

Metrics computed (all at K=5):
  hit@5        — did at least 1 ground-truth product appear in top 5?  (0 or 1)
  precision@5  — fraction of top-5 results that are ground truth        (0–1)
  recall@5     — fraction of ground-truth items found in top 5          (0–1)
  ndcg@5       — normalized discounted cumulative gain                  (0–1)
  mrr          — reciprocal rank of the first correct match             (0–1)

Usage:
  cd mlops
  python evaluate_mlflow.py

  # Override defaults:
  GATEWAY_URL=http://localhost:8000 \\
  MLFLOW_TRACKING_URI=http://localhost:5000 \\
  python evaluate_mlflow.py

Requirements:
  pip install mlflow requests
"""

import base64
import io
import json
import math
import os
import time
import urllib.request
import urllib.error

import requests
import mlflow

# ─── Configuration ────────────────────────────────────────────────────────────
GATEWAY_URL         = os.getenv("GATEWAY_URL",         "http://localhost:8000")
GOLDEN_DATASET_PATH = os.getenv("GOLDEN_DATASET_PATH", "golden_dataset.json")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME     = "locus_search_accuracy"
TOP_K               = 5
# ──────────────────────────────────────────────────────────────────────────────

STALE_SOURCES = {"full_image", "unknown", None, ""}


def load_golden_dataset(path: str) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Golden dataset not found at '{path}'.\n"
            "Run  python annotate_golden.py  or  python build_golden_from_lens.py  first."
        )
    with open(path) as f:
        data = json.load(f)
    if not data:
        raise ValueError("Golden dataset is empty — annotate some products first.")
    return data


def get_image_bytes(url: str) -> bytes:
    """
    Fetch image bytes from either an https URL or a data: URI.

    data: URIs look like:  data:image/jpeg;base64,/9j/4AAQ...
    These were accidentally stored when a query image was dragged into
    the builder instead of pasting a real URL. The pixel data is intact —
    we just decode the base64 directly instead of making an HTTP request.
    """
    if url.startswith("data:"):
        _, b64data = url.split(",", 1)
        return base64.b64decode(b64data)

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; Locus-Evaluator/1.0)"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read()


def run_search(image_bytes: bytes, filename: str = "query.jpg") -> list:
    files    = {"file": (filename, io.BytesIO(image_bytes), "image/jpeg")}
    response = requests.post(f"{GATEWAY_URL}/search", files=files, timeout=30)
    response.raise_for_status()
    body = response.json()
    if isinstance(body, dict) and "matches" in body:
        return body["matches"]
    if isinstance(body, list):
        return body
    return []


def extract_product_ids(matches: list) -> list:
    ids = []
    for m in matches:
        pid = m.get("product_id") or m.get("item_id") or m.get("id")
        if pid:
            ids.append(str(pid))
    return ids


def compute_metrics(retrieved_ids: list, relevant_ids: list, k: int = TOP_K) -> dict:
    retrieved_at_k = retrieved_ids[:k]
    relevant_set   = set(str(r) for r in relevant_ids)
    n_relevant     = len(relevant_set)

    hit_at_k       = int(any(rid in relevant_set for rid in retrieved_at_k))
    n_correct      = sum(1 for rid in retrieved_at_k if rid in relevant_set)
    precision_at_k = n_correct / k
    recall_at_k    = n_correct / n_relevant if n_relevant > 0 else 0.0

    mrr = 0.0
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            mrr = 1.0 / rank
            break

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, rid in enumerate(retrieved_at_k, start=1)
        if rid in relevant_set
    )
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(n_relevant, k) + 1)
    )
    ndcg_at_k = dcg / idcg if idcg > 0 else 0.0

    return {
        f"hit@{k}"          : hit_at_k,
        f"precision@{k}"    : precision_at_k,
        f"recall@{k}"       : recall_at_k,
        "mrr"               : mrr,
        f"ndcg@{k}"         : ndcg_at_k,
        "n_correct_in_top_k": n_correct,
    }


def run_evaluation(dataset: list, run_name: str = None) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n  MLFlow run ID  : {run.info.run_id}")
        print(f"  Experiment     : {EXPERIMENT_NAME}\n")

        mlflow.log_param("num_queries",    len(dataset))
        mlflow.log_param("top_k",          TOP_K)
        mlflow.log_param("gateway_url",    GATEWAY_URL)
        mlflow.log_param("golden_dataset", GOLDEN_DATASET_PATH)

        n_stale    = sum(1 for e in dataset if e.get("query_box_source") in STALE_SOURCES)
        n_data_uri = sum(1 for e in dataset if e.get("query_image_url", "").startswith("data:"))
        mlflow.log_param("queries_with_clean_crop", len(dataset) - n_stale)
        mlflow.log_param("queries_with_stale_crop", n_stale)
        mlflow.log_param("queries_with_data_uri",   n_data_uri)

        if n_data_uri > 0:
            print(f"  ℹ️  {n_data_uri}/{len(dataset)} queries use embedded data: images")
            print(f"     (decoding base64 directly — no HTTP download needed).\n")

        mlflow.log_artifact(GOLDEN_DATASET_PATH, artifact_path="golden_dataset")

        per_query_results = []
        accumulator = {
            f"hit@{TOP_K}"      : [],
            f"precision@{TOP_K}": [],
            f"recall@{TOP_K}"   : [],
            "mrr"               : [],
            f"ndcg@{TOP_K}"     : [],
        }

        for step, entry in enumerate(dataset):
            query_product_id = entry.get("query_product_id") or entry.get("query_name", f"query_{step}")
            image_url        = entry["query_image_url"]
            relevant_ids     = entry["relevant_product_ids"]
            label            = entry.get("query_name") or query_product_id
            is_data          = image_url.startswith("data:")

            print(f"  [{step + 1:3d}/{len(dataset)}]  {label[:50]}" + (" [embedded]" if is_data else ""))

            try:
                t0          = time.time()
                image_bytes = get_image_bytes(image_url)
                raw_matches = run_search(image_bytes)
                latency_ms  = (time.time() - t0) * 1000

                retrieved_ids = extract_product_ids(raw_matches)

                if not retrieved_ids:
                    print("            ⚠️  Search returned no results")

                metrics = compute_metrics(retrieved_ids, relevant_ids, k=TOP_K)
                metrics["latency_ms"] = latency_ms

                correct = metrics["n_correct_in_top_k"]
                print(
                    f"            ✅  {correct}/{TOP_K} correct  |  "
                    f"prec={metrics[f'precision@{TOP_K}']:.2f}  "
                    f"ndcg={metrics[f'ndcg@{TOP_K}']:.2f}  "
                    f"mrr={metrics['mrr']:.2f}  "
                    f"latency={latency_ms:.0f}ms"
                )

                mlflow.log_metrics({f"q/{k}".replace("@", "_at_"): v for k, v in metrics.items()},step=step,)

                per_query_results.append({
                    "query_product_id": query_product_id,
                    "query_name"      : label,
                    "relevant_ids"    : relevant_ids,
                    "retrieved_ids"   : retrieved_ids[:TOP_K],
                    "query_src"       : "data_uri" if is_data else "url",
                    **metrics,
                })

                for key in accumulator:
                    accumulator[key].append(metrics[key])

            except urllib.error.URLError as e:
                print(f"            ❌  Image download failed: {e}")
                per_query_results.append({"query_product_id": query_product_id, "error": f"download: {e}"})
            except requests.HTTPError as e:
                print(f"            ❌  Search request failed: {e}")
                per_query_results.append({"query_product_id": query_product_id, "error": f"search: {e}"})
            except Exception as e:
                print(f"            ❌  Unexpected error: {e}")
                per_query_results.append({"query_product_id": query_product_id, "error": str(e)})

        # Aggregate
        avg_metrics = {}
        for key, values in accumulator.items():
            if values:
                avg_metrics[f"avg_{key}"] = sum(values) / len(values)

        n_failed = sum(1 for r in per_query_results if "error" in r)
        avg_metrics["n_successful"] = len(dataset) - n_failed
        avg_metrics["n_failed"]     = n_failed

        mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in avg_metrics.items()})

        results_path = "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(per_query_results, f, indent=2)
        mlflow.log_artifact(results_path)

        print()
        print("  " + "═" * 54)
        print("  📊  EVALUATION SUMMARY")
        print("  " + "═" * 54)
        print(f"  {'Queries':<30} {len(dataset) - n_failed}/{len(dataset)} successful")
        print()
        for key, val in avg_metrics.items():
            if key.startswith("avg_"):
                label = key.replace("avg_", "").replace("@", " @ ").replace("_", " ")
                bar   = "█" * int(val * 20) + "░" * (20 - int(val * 20))
                print(f"  {label:<30} {val:.4f}  {bar}")
        print("  " + "═" * 54)

        return avg_metrics


def main():
    print()
    print("═" * 64)
    print("  🔍  Locus MLFlow Evaluator")
    print("═" * 64)
    print(f"  Golden dataset  : {GOLDEN_DATASET_PATH}")
    print(f"  Gateway         : {GATEWAY_URL}")
    print(f"  MLFlow server   : {MLFLOW_TRACKING_URI}")
    print(f"  Experiment      : {EXPERIMENT_NAME}")
    print(f"  Top-K           : {TOP_K}")
    print()

    dataset = load_golden_dataset(GOLDEN_DATASET_PATH)
    print(f"  Loaded {len(dataset)} annotated queries.\n")

    run_name = input("  Run name (press Enter for auto-timestamp): ").strip() or None
    run_evaluation(dataset, run_name)

    print()
    print(f"  ✅  Done. Open http://localhost:5000 to see results in MLFlow.")
    print("═" * 64)
    print()


if __name__ == "__main__":
    main()