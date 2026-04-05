# -*- coding: utf-8 -*-
"""
Gemini Visual Similarity Judge — Locus
=======================================

Uses Google Gemini Vision (free tier) to judge how visually similar
your search results are to the query image.

METRIC: VSS@5 (Visual Similarity Score at 5)
----------------------------------------------
  For each query:
    1. Send the query image to /search → get top 5 results
    2. For each result, show Gemini BOTH images and ask:
       "How visually similar are these two fashion items? (0-10)"
    3. Average the 5 scores → VSS@5  (normalised to 0.0–1.0)

  VSS@5 is your headline metric. Unlike ACS@5 (pure vector math),
  Gemini understands fashion: style, silhouette, colour, cut, fabric.
  This makes VSS@5 a much more human-aligned quality signal.

  VSS@5 interpretation:
    >= 0.80  Excellent — results look almost identical to the query
    >= 0.65  Good      — strong visual match
    >= 0.50  Fair      — loosely related
     < 0.50  Poor      — results don't match visually

Setup:
  1. Get a free Gemini API key at https://aistudio.google.com/apikey
  2. Open the .env file in the locus root folder and set:
       GEMINI_API_KEY=your_key_here
  3. Install the dependency (one-time):
       pip install google-generativeai

Usage (PowerShell):
  cd "C:\\Users\\User\\Desktop\\Locus Project\\locus\\mlops"
  python -u evaluate_gemini_judge.py

Requirements:
  pip install google-generativeai requests mlflow Pillow
"""

import base64
import io
import json
import os
import pathlib
import re
import sys
import time
import urllib.request
import urllib.error

import requests
import mlflow

# ── Load .env from the locus root folder (one level up from mlops/) ───────────
_env_path = pathlib.Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip().strip('"').strip("'"))

# ── Configuration ─────────────────────────────────────────────────────────────
GATEWAY_URL         = os.getenv("GATEWAY_URL",         "http://localhost:8000")
GOLDEN_DATASET_PATH = os.getenv("GOLDEN_DATASET_PATH", "golden_dataset.json")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY",      "")
EXPERIMENT_NAME     = "locus_gemini_judge"
TOP_K               = 5
GEMINI_MODEL        = "gemini-2.0-flash"    # free tier: 15 req/min
SLEEP_BETWEEN_CALLS = 4.5                  # seconds — stay under rate limit
# ──────────────────────────────────────────────────────────────────────────────

JUDGE_PROMPT = (
    "You are a fashion visual similarity expert.\n"
    "I will show you two fashion product images.\n"
    "The FIRST image is the QUERY (what the shopper is looking for).\n"
    "The SECOND image is a SEARCH RESULT returned by the system.\n\n"
    "Rate how visually similar the search result is to the query "
    "on a scale from 0 to 10:\n"
    "  10 = identical item (same colour, cut, style, fabric)\n"
    "   8 = very similar (minor colour or detail differences)\n"
    "   6 = similar category and style, noticeable differences\n"
    "   4 = same broad category but clearly different style\n"
    "   2 = loosely related (e.g. both are tops, but very different)\n"
    "   0 = completely unrelated\n\n"
    "Reply with a SINGLE integer from 0 to 10. Nothing else."
)


def p(msg=""):
    """Print with immediate flush so PowerShell shows output in real time."""
    print(msg, flush=True)


# ── Gemini setup ──────────────────────────────────────────────────────────────

def init_gemini():
    """Initialise the Gemini client. Exits cleanly if the key is missing."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        p()
        p("  [error] GEMINI_API_KEY is not set.")
        p("  Open the .env file in the locus root folder and add:")
        p("    GEMINI_API_KEY=your_key_here")
        p("  Get a free key at: https://aistudio.google.com/apikey")
        p()
        sys.exit(1)

    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        return model, genai
    except ImportError:
        p()
        p("  [error] google-generativeai is not installed.")
        p("  Run:  pip install google-generativeai")
        p()
        sys.exit(1)


# ── Image helpers ─────────────────────────────────────────────────────────────

def get_image_bytes(url: str) -> bytes:
    """
    Fetch image bytes from a URL or a base64 data URI.
    The golden_dataset stores query images as data URIs (data:image/jpeg;base64,...).
    Search result images are plain HTTPS URLs.
    """
    if url.startswith("data:"):
        _, b64data = url.split(",", 1)
        return base64.b64decode(b64data)

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; Locus-GeminiJudge/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read()


def bytes_to_pil(image_bytes: bytes):
    """Convert raw bytes to a PIL Image (required by google-generativeai)."""
    from PIL import Image
    return Image.open(io.BytesIO(image_bytes))


# ── Search helper ─────────────────────────────────────────────────────────────

def run_search(image_bytes: bytes) -> list:
    """POST the query image to /search and return a list of result objects."""
    files    = {"file": ("query.jpg", io.BytesIO(image_bytes), "image/jpeg")}
    response = requests.post(f"{GATEWAY_URL}/search", files=files, timeout=30)
    response.raise_for_status()
    body = response.json()

    # Support multiple response shapes from the gateway
    if isinstance(body, dict) and "matches" in body:
        return body["matches"]
    if isinstance(body, list):
        return body
    for key in ("results", "items"):
        if isinstance(body, dict) and key in body:
            return body[key]
    return []


# ── Gemini judge ──────────────────────────────────────────────────────────────

def judge_pair(model, query_pil, result_pil) -> float | None:
    """
    Ask Gemini to rate the visual similarity between a query and a result image.
    Returns a float in [0.0, 1.0], or None if the call fails.
    """
    try:
        response = model.generate_content(
            [query_pil, result_pil, JUDGE_PROMPT],
            generation_config={"temperature": 0, "max_output_tokens": 8},
        )
        raw = response.text.strip()

        # Parse the integer out of the response (handles "8", "8/10", "Score: 8", etc.)
        match = re.search(r"\b(\d+)\b", raw)
        if match:
            score_10 = int(match.group(1))
            score_10 = max(0, min(10, score_10))   # clamp to [0, 10]
            return round(score_10 / 10.0, 4)
        p(f"            [warn] Could not parse Gemini response: '{raw}'")
        return None

    except Exception as e:
        p(f"            [error] Gemini call failed: {e}")
        return None


# ── Dataset loader ────────────────────────────────────────────────────────────

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


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_evaluation(model, dataset: list, run_name: str = None) -> dict:

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
        mlflow.log_param("gemini_model",   GEMINI_MODEL)
        mlflow.log_param("golden_dataset", GOLDEN_DATASET_PATH)
        mlflow.log_param("metric",         "gemini_visual_similarity")
        mlflow.log_artifact(GOLDEN_DATASET_PATH, artifact_path="golden_dataset")

        all_vss5      = []
        all_top1      = []
        per_query_log = []

        for step, entry in enumerate(dataset):
            label   = entry.get("query_name") or entry.get("query_product_id") or f"query_{step}"
            img_url = entry["query_image_url"]
            is_data = img_url.startswith("data:")

            tag = " [embedded]" if is_data else ""
            p(f"\n  [{step + 1:3d}/{len(dataset)}]  {label[:55]}{tag}")

            try:
                # ── 1. Fetch query image ───────────────────────────────────
                query_bytes = get_image_bytes(img_url)
                query_pil   = bytes_to_pil(query_bytes)

                # ── 2. Run search ──────────────────────────────────────────
                p("            [search] Running search ...")
                t0         = time.time()
                results    = run_search(query_bytes)
                latency_ms = (time.time() - t0) * 1000
                results_k  = results[:TOP_K]
                p(f"            [ok] Got {len(results_k)} results in {latency_ms:.0f}ms")

                if not results_k:
                    p("            [warn] No results — skipping")
                    per_query_log.append({"label": label, "error": "no results"})
                    continue

                # ── 3. Gemini judges each result ───────────────────────────
                scores = []
                for rank, result in enumerate(results_k, start=1):
                    result_img_url = (
                        result.get("image_url")
                        or result.get("payload", {}).get("image_url", "")
                    )
                    if not result_img_url:
                        p(f"            [{rank}] [warn] No image_url in result — skipping")
                        continue

                    try:
                        result_bytes = get_image_bytes(result_img_url)
                        result_pil   = bytes_to_pil(result_bytes)
                    except Exception as e:
                        p(f"            [{rank}] [warn] Could not fetch result image: {e}")
                        continue

                    p(f"            [{rank}] Asking Gemini ...")
                    score = judge_pair(model, query_pil, result_pil)
                    time.sleep(SLEEP_BETWEEN_CALLS)   # respect free-tier rate limit

                    if score is not None:
                        scores.append(score)
                        bar = "#" * int(score * 20) + "." * (20 - int(score * 20))
                        result_name = result.get("name", "")[:40]
                        p(f"            [{rank}] VSS={score:.2f}  [{bar}]  {result_name}")
                    else:
                        p(f"            [{rank}] [warn] No score returned")

                if not scores:
                    p("            [warn] No scores collected — skipping query")
                    per_query_log.append({"label": label, "error": "no scores"})
                    continue

                # ── 4. Aggregate ───────────────────────────────────────────
                vss5 = sum(scores) / len(scores)
                top1 = scores[0]

                all_vss5.append(vss5)
                all_top1.append(top1)

                p(f"\n            [stats] VSS@5 = {vss5:.4f}   Top-1 score = {top1:.4f}")

                mlflow.log_metrics({
                    "q/vss5":       vss5,
                    "q/top1_score": top1,
                    "q/latency_ms": latency_ms,
                }, step=step)

                per_query_log.append({
                    "label":      label,
                    "vss5":       vss5,
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
                p(f"            [error] Unexpected error: {e}")
                per_query_log.append({"label": label, "error": str(e)})

        # ── Aggregate across all queries ───────────────────────────────────
        n_ok = len(all_vss5)
        if n_ok == 0:
            p("\n  [warn] No queries could be evaluated.")
            p("  Make sure docker compose is running:  docker compose up -d")
            return {}

        headline = {
            "avg_vss5":       round(sum(all_vss5) / n_ok, 4),
            "avg_top1_score": round(sum(all_top1) / n_ok, 4),
            "n_evaluated":    n_ok,
            "n_failed":       len(dataset) - n_ok,
        }
        mlflow.log_metrics(headline)

        results_path = "gemini_judge_results.json"
        with open(results_path, "w") as f:
            json.dump(per_query_log, f, indent=2)
        mlflow.log_artifact(results_path)

        def bar(v):
            return "#" * int(v * 20) + "." * (20 - int(v * 20))

        vss5_val = headline["avg_vss5"]
        top1_val = headline["avg_top1_score"]

        p()
        p("  " + "=" * 58)
        p("  GEMINI VISUAL SIMILARITY JUDGE — SUMMARY")
        p("  " + "=" * 58)
        p(f"  Queries evaluated  : {n_ok} / {len(dataset)}")
        p()
        p(f"  VSS@5  (headline)  : {vss5_val:.4f}  [{bar(vss5_val)}]")
        p(f"  Avg top-1 score    : {top1_val:.4f}  [{bar(top1_val)}]")
        p()
        p("  What VSS@5 means:")
        p("    >= 0.80  Excellent — results look almost identical to query")
        p("    >= 0.65  Good      — strong visual match")
        p("    >= 0.50  Fair      — loosely related")
        p("     < 0.50  Poor      — results don't match visually")
        p()
        p("  Run this before and after each model change to track improvement.")
        p("  View results in MLflow UI:")
        p("    mlflow ui --backend-store-uri sqlite:///mlflow.db")
        p("    Then open: http://localhost:5000")
        p("  " + "=" * 58)

        return headline


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p()
    p("=" * 64)
    p("  Locus — Gemini Visual Similarity Judge")
    p(f"  Model  : {GEMINI_MODEL}")
    p(f"  Metric : VSS@5  (Gemini-rated, 0.0 – 1.0)")
    p("=" * 64)
    p(f"  Dataset        : {GOLDEN_DATASET_PATH}")
    p(f"  Gateway        : {GATEWAY_URL}")
    p(f"  MLflow storage : {MLFLOW_TRACKING_URI}")
    p(f"  Top-K          : {TOP_K}")
    p()

    model, _ = init_gemini()
    p("  [ok] Gemini client ready.")
    p()

    dataset = load_golden_dataset(GOLDEN_DATASET_PATH)
    n_data  = sum(1 for e in dataset if e.get("query_image_url", "").startswith("data:"))
    p(f"  Loaded {len(dataset)} queries  ({n_data} embedded, {len(dataset) - n_data} URLs)")
    p()

    p("  Make sure Docker is running before continuing:")
    p("    docker compose up -d   (run from the locus root folder)")
    p()
    p(f"  Note: Each result image is sent to Gemini separately.")
    p(f"  Expected time: ~{len(dataset) * TOP_K * SLEEP_BETWEEN_CALLS / 60:.1f} minutes")
    p("  (free tier rate limit: 15 req/min — sleeping {:.1f}s between calls)".format(SLEEP_BETWEEN_CALLS))
    p()

    run_name = input("  Run name (press Enter for auto-timestamp): ").strip() or None
    confirm  = input(f"  Start evaluation? [y/N]: ").strip().lower()
    if confirm != "y":
        p("  Aborted.")
        return

    headline = run_evaluation(model, dataset, run_name)

    if headline:
        p()
        p("  [ok] Done! Results saved to gemini_judge_results.json")
    p("=" * 64)
    p()


if __name__ == "__main__":
    main()
