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
import hashlib
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
GEMINI_MODEL        = "gemini-2.5-flash"        # updated: 2.0-flash-lite deprecated June 2026
SLEEP_BETWEEN_CALLS = 4.5                      # seconds — safe under 15 RPM free-tier limit
MAX_RETRIES         = 4                        # retries on 429 rate-limit errors
VSS_CACHE_PATH      = os.getenv("VSS_CACHE_PATH", "vss_cache.json")
# ──────────────────────────────────────────────────────────────────────────────

JUDGE_PROMPT = (
    "You are a fashion visual similarity expert.\n"
    "I will show you two fashion product images.\n"
    "The FIRST image is the QUERY (what the shopper is looking for).\n"
    "The SECOND image is a SEARCH RESULT returned by the system.\n\n"
    "Rate how visually similar the search result is to the query.\n"
    "Give a score from 0.00 to 1.00 using exactly two decimal places.\n\n"
    "Anchor points for calibration:\n"
    "  1.00 = identical item (same colour, cut, style, fabric)\n"
    "  0.80 = very similar — minor colour or detail difference\n"
    "  0.60 = similar style and category, but noticeable differences\n"
    "  0.40 = same broad category, clearly different cut or style\n"
    "  0.20 = loosely related (e.g. both are tops but very different)\n"
    "  0.00 = completely unrelated\n\n"
    "You MUST interpolate between these anchors to use the full range.\n"
    "Examples of precise scores: 0.73, 0.55, 0.91, 0.38, 0.67\n\n"
    "Reply with ONLY a decimal number between 0.00 and 1.00. Nothing else."
)


def p(msg=""):
    """Print with immediate flush so PowerShell shows output in real time."""
    print(msg, flush=True)


# ── VSS Cache ─────────────────────────────────────────────────────────────────

class VSSCache:
    """
    Persistent on-disk cache for VSS scores.

    Stores every scored (query, result) pair in a local JSON file so Gemini
    is NEVER called twice for the same pair — even across separate eval runs.

    Key  : SHA-256( query_image_url + "|" + result_image_url )[:24]
    Value: { vss, model, ts }
    """

    def __init__(self, path: str = "vss_cache.json"):
        self._path   = path
        self._data   = {}
        self._hits   = 0
        self._misses = 0
        if os.path.exists(path):
            try:
                with open(path) as f:
                    self._data = json.load(f)
            except Exception as e:
                p(f"  [warn] Could not load VSS cache ({e}). Starting fresh.")

    def _key(self, query_url: str, result_url: str) -> str:
        return hashlib.sha256(
            f"{query_url}|{result_url}".encode("utf-8")
        ).hexdigest()[:24]

    def lookup(self, query_url: str, result_url: str):
        """Return cached vss score (float) if pair already scored, else None."""
        entry = self._data.get(self._key(query_url, result_url))
        if entry is None:
            self._misses += 1
            return None
        self._hits += 1
        return entry["vss"]

    def store(self, query_url: str, result_url: str, vss: float, model: str):
        """Persist a freshly-scored pair to disk immediately."""
        self._data[self._key(query_url, result_url)] = {
            "vss":   round(vss, 4),
            "model": model,
            "ts":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        try:
            with open(self._path, "w") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            p(f"  [warn] Could not save VSS cache: {e}")

    @property
    def size(self)  -> int: return len(self._data)
    @property
    def hits(self)  -> int: return self._hits
    @property
    def misses(self)-> int: return self._misses


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
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        return client
    except ImportError:
        p()
        p("  [error] google-genai is not installed.")
        p("  Run:  pip install google-genai")
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


# ── Search helper ─────────────────────────────────────────────────────────────

# Mirrors CATEGORY_ALIASES in vectorizer.py — used for bbox tier-2 matching.
CATEGORY_ALIASES = {
    "top":        ["top", "shirt"],
    "sports_bra": ["sports_bra", "top", "shirt"],
    "sweater":    ["sweater", "top", "shirt"],
    "jacket":     ["jacket", "coat"],
    "dress":      ["dress", "jumpsuit"],
    "jumpsuit":   ["jumpsuit", "dress"],
    "skirt":      ["skirt", "dress"],
    "pants":      ["pants", "trousers", "jeans"],
    "leggings":   ["leggings", "pants", "shorts"],
    "shorts":     ["shorts", "pants"],
    "shoes":      ["shoes", "sneakers", "boots", "sandals"],
    "hat":        ["hat", "cap"],
    "bag":        ["bag", "handbag", "backpack"],
}


def _select_box(detections: list, category_tag: str):
    """
    Pick the best detection for a query using the known category_tag.
      T1 — exact search_label match
      T2 — alias match (CATEGORY_ALIASES)
    No T3: a wrong-category crop is worse than full image + correct filter.
    When T1+T2 fail, caller falls back to full image + category_tag filter.
    Returns (detection_dict, tier_str) or (None, None).
    """
    if not detections:
        return None, None

    # T1 — exact match
    exact = [d for d in detections if d.get("search_label") == category_tag]
    if exact:
        return max(exact, key=lambda d: d.get("score", 0)), "T1-exact"

    # T2 — alias match
    aliases = CATEGORY_ALIASES.get(category_tag, [category_tag])
    alias_hits = [d for d in detections if d.get("search_label") in aliases]
    if alias_hits:
        return max(alias_hits, key=lambda d: d.get("score", 0)), "T2-alias"

    return None, None


def run_search(image_bytes: bytes, category_tag: str = "") -> list:
    """
    1. POST to /detect — get YOLO bounding boxes + CLIP-relabelled search_labels.
    2. Select the best bbox using the known category_tag (mirrors indexing tier logic).
    3. POST to /search with bbox coords + search_label (or full image if no box found).
    """
    # Step 1: detect
    detect_resp = requests.post(
        f"{GATEWAY_URL}/detect",
        files={"file": ("query.jpg", io.BytesIO(image_bytes), "image/jpeg")},
        timeout=30,
    )
    detect_resp.raise_for_status()
    detections = detect_resp.json().get("detections", [])

    # Step 2: select box
    selected, tier = _select_box(detections, category_tag) if category_tag else (None, None)

    if selected:
        x1, y1, x2, y2 = selected["bbox"]
        sl = selected.get("search_label", "")
        sc = selected.get("score", 0)
        p(f"            [bbox] {tier}  label={sl}  score={sc:.2f}  box=({x1},{y1},{x2},{y2})")
        data = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "search_label": sl}
    else:
        reason = "no detections" if not detections else f"no match for '{category_tag}'"
        p(f"            [bbox] none ({reason}) — using full image")
        data = {}

    # Step 3: search
    files    = {"file": ("query.jpg", io.BytesIO(image_bytes), "image/jpeg")}
    response = requests.post(f"{GATEWAY_URL}/search", files=files, data=data, timeout=30)
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

def judge_pair(client, query_bytes: bytes, result_bytes: bytes) -> float | None:
    """
    Ask Gemini to rate the visual similarity between a query and a result image.
    Returns a float in [0.0, 1.0], or None if the call fails.
    Retries automatically on 429 rate-limit errors using the wait time from the error.
    """
    from google.genai import types

    contents = [
        types.Part.from_bytes(data=query_bytes,  mime_type="image/jpeg"),
        types.Part.from_bytes(data=result_bytes, mime_type="image/jpeg"),
        JUDGE_PROMPT,
    ]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=32,                              # enough for "0.73" or "1.00"
                    thinking_config=types.ThinkingConfig(thinking_budget=0),  # disable reasoning — we just need a digit
                ),
            )

            # response.text is None when safety filters block the output
            if response.text is None:
                reason = "unknown"
                try:
                    reason = str(response.candidates[0].finish_reason)
                except Exception:
                    pass
                p(f"            [warn] Gemini returned no text (blocked — reason: {reason}). Skipping pair.")
                return None

            raw = response.text.strip()

            # Parse a float in [0, 1] — handles "0.73", "0.7", "1.0", "1"
            # Fallback: if Gemini ignores the instruction and returns an integer
            #   (e.g. "8"), treat it as a 0-10 score and divide by 10.
            match = re.search(r"\b(\d+(?:\.\d+)?)\b", raw)
            if match:
                val = float(match.group(1))
                if val > 1.0:
                    # Gemini returned an integer scale — normalise gracefully
                    val = val / 10.0 if val <= 10.0 else val / 100.0
                score = max(0.0, min(1.0, val))
                return round(score, 2)
            p(f"            [warn] Could not parse Gemini response: '{raw}'")
            return None

        except Exception as e:
            err = str(e)
            is_rate_limit  = "429" in err
            is_transient   = "503" in err or "500" in err or "UNAVAILABLE" in err.upper()

            if is_rate_limit:
                # 429: respect the retry-after header if present, else wait 60s
                wait_match = re.search(r"retry in (\d+(?:\.\d+)?)s", err, re.IGNORECASE)
                wait = float(wait_match.group(1)) + 2 if wait_match else 60.0
                p(f"            [rate-limit] Attempt {attempt}/{MAX_RETRIES} — waiting {wait:.0f}s ...")
                time.sleep(wait)
            elif is_transient:
                # 503 / 500: server overloaded — exponential backoff then retry
                wait = (2 ** attempt) * 5.0   # 10s, 20s, 40s ...
                p(f"            [transient-error] Attempt {attempt}/{MAX_RETRIES} — {err[:80]} — retrying in {wait:.0f}s ...")
                time.sleep(wait)
            else:
                # Permanent error (bad request, auth failure, etc.) — no point retrying
                p(f"            [error] Gemini call failed: {e}")
                return None

            if attempt == MAX_RETRIES:
                p(f"            [error] Giving up after {MAX_RETRIES} attempts.")
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

def run_evaluation(client, dataset: list, run_name: str = None) -> dict:

    # -- Initialise VSS cache --------------------------------------------------
    cache = VSSCache(VSS_CACHE_PATH)
    p(f"  VSS cache: {cache.size} entries loaded from '{VSS_CACHE_PATH}'")

    p("  Connecting to MLflow ...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    p("  MLflow ready.")

    with mlflow.start_run(run_name=run_name) as run:
        p(f"\n  MLflow run ID : {run.info.run_id}")
        p(f"  Experiment    : {EXPERIMENT_NAME}\n")

        mlflow.log_param("num_queries",       len(dataset))
        mlflow.log_param("top_k",             TOP_K)
        mlflow.log_param("gateway_url",       GATEWAY_URL)
        mlflow.log_param("gemini_model",      GEMINI_MODEL)
        mlflow.log_param("golden_dataset",    GOLDEN_DATASET_PATH)
        mlflow.log_param("metric",            "gemini_visual_similarity")
        mlflow.log_param("vss_cache_path",    VSS_CACHE_PATH)
        mlflow.log_param("vss_cache_preloaded", cache.size)
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

                # ── 2. Run search ──────────────────────────────────────────
                category_tag = entry.get("query_category_tag", "")
                p(f"            [search] category={category_tag or '(none)'}  — detecting bbox ...")
                t0         = time.time()
                results    = run_search(query_bytes, category_tag)
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

                    # -- Cache lookup: skip Gemini if pair already scored ------
                    cached_score = cache.lookup(img_url, result_img_url)
                    if cached_score is not None:
                        scores.append(cached_score)
                        bar = "#" * int(cached_score * 20) + "." * (20 - int(cached_score * 20))
                        p(f"            [{rank}] VSS={cached_score:.2f}  [{bar}]  [CACHE HIT]")
                        continue   # no API call, no rate-limit sleep needed

                    # -- Cache miss: fetch image and call Gemini ---------------
                    try:
                        result_bytes = get_image_bytes(result_img_url)
                    except Exception as e:
                        p(f"            [{rank}] [warn] Could not fetch result image: {e}")
                        continue

                    p(f"            [{rank}] Asking Gemini ...")
                    score = judge_pair(client, query_bytes, result_bytes)
                    time.sleep(SLEEP_BETWEEN_CALLS)   # respect free-tier rate limit (real calls only)

                    if score is not None:
                        scores.append(score)
                        cache.store(img_url, result_img_url, score, GEMINI_MODEL)
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
        mlflow.log_metrics({
            "cache/hits":   cache.hits,
            "cache/misses": cache.misses,
            "cache/size":   cache.size,
        })

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
        p()
        p(f"  VSS cache: {cache.size} total entries")
        p(f"    Cache hits  (free) : {cache.hits}")
        p(f"    Cache misses (API) : {cache.misses}")
        p()
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

    client = init_gemini()
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

    headline = run_evaluation(client, dataset, run_name)

    if headline:
        p()
        p("  [ok] Done! Results saved to gemini_judge_results.json")
    p("=" * 64)
    p()


if __name__ == "__main__":
    main()
