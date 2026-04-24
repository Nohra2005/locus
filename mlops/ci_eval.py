"""
ci_eval.py — Judge-based CI quality gate.

Calls /search on 5 diverse golden queries (local images, no external deps),
judges the top-3 results per query with Gemini via OpenRouter.
Exits 1 if the overall average score falls below MIN_SCORE (default 0.65).

Usage:
    python mlops/ci_eval.py --gateway-url http://localhost:8000
    python mlops/ci_eval.py --gateway-url http://localhost:8000 --min-score 0.70
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
import threading
from pathlib import Path

import requests

SCRIPT_DIR          = Path(__file__).parent
GOLDEN_DATASET_PATH = SCRIPT_DIR / "golden_dataset.json"
GOLDEN_IMAGES_DIR   = SCRIPT_DIR / "golden_images"
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
_OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
_MODEL              = "google/gemini-2.0-flash-001"

# 5 diverse queries — one per major category, all have local images in golden_images/
EVAL_QUERIES = [
    "blue high waisted straight leg jeans",    # pants
    "black satin slip midi dress",             # dress
    "white leather chunky sneakers",           # shoes
    "camel longline wool coat",                # jacket
    "black shoulder bag",                      # bag
]

_JUDGE_PROMPT = (
    "You are a fashion visual similarity expert.\n"
    "Rate how visually similar the result image is to the query image on a scale 0.00–1.00.\n"
    "0.00 = completely unrelated, 1.00 = identical item.\n"
    "Respond with ONLY the numeric score. No explanation."
)

_lock      = threading.Lock()
_last_call = 0.0


def _judge(query_b64: str, result_url: str, gateway_url: str) -> float | None:
    global _last_call
    with _lock:
        gap = _last_call + 2.0 - time.monotonic()
        if gap > 0:
            time.sleep(gap)
        _last_call = time.monotonic()

    # Rewrite localhost URLs to the actual gateway for CI
    result_url = result_url.replace("http://localhost:8000", gateway_url)

    try:
        r = requests.get(result_url, timeout=10)
        r.raise_for_status()
        result_b64 = base64.b64encode(r.content).decode()
    except Exception as e:
        print(f"    [WARN] Could not fetch result image: {e}")
        return None

    payload = {
        "model": _MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text",      "text": _JUDGE_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{query_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{result_b64}"}},
        ]}],
        "max_tokens": 10,
        "temperature": 0.0,
    }
    try:
        r = requests.post(
            _OPENROUTER_URL,
            json=payload,
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            timeout=30,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        m = re.search(r"\d+\.\d+|\d+", text)
        return float(m.group()) if m else None
    except Exception as e:
        print(f"    [WARN] Judge call failed: {e}")
        return None


def run_eval(gateway_url: str) -> tuple[float, int]:
    dataset       = json.loads(GOLDEN_DATASET_PATH.read_text())
    name_to_entry = {e.get("query_name", ""): e for e in dataset}
    all_scores: list[float] = []
    evaluated = 0

    for query_name in EVAL_QUERIES:
        entry = name_to_entry.get(query_name)
        if not entry:
            print(f"  [SKIP] '{query_name}' not in golden dataset")
            continue

        img_url  = entry.get("query_image_url", "")
        fname    = img_url.rsplit("/", 1)[-1]
        img_path = GOLDEN_IMAGES_DIR / fname
        if not img_path.exists():
            print(f"  [SKIP] local image not found: {img_path.name}")
            continue

        image_bytes = img_path.read_bytes()
        query_b64   = base64.b64encode(image_bytes).decode()
        category    = entry.get("query_category_tag", "")

        try:
            r = requests.post(
                f"{gateway_url}/search",
                files={"file": ("q.jpg", io.BytesIO(image_bytes), "image/jpeg")},
                data={"search_label": category} if category else {},
                timeout=30,
            )
            r.raise_for_status()
            matches = r.json().get("matches", [])[:3]
        except Exception as e:
            print(f"  [FAIL] /search error for '{query_name}': {e}")
            continue

        if not matches:
            print(f"  [WARN] no results returned for '{query_name}'")
            continue

        scores = []
        for m in matches:
            result_img_url = m.get("image_url", "")
            if not result_img_url:
                continue
            score = _judge(query_b64, result_img_url, gateway_url)
            if score is not None:
                scores.append(score)

        if scores:
            avg = sum(scores) / len(scores)
            print(f"  '{query_name}' [{category}]: {len(scores)} scores, avg={avg:.3f}")
            all_scores.extend(scores)
            evaluated += 1
        else:
            print(f"  [WARN] all judge calls failed for '{query_name}'")

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return overall, evaluated


def main():
    parser = argparse.ArgumentParser(description="CI judge-based quality gate for Locus")
    parser.add_argument("--gateway-url", default="http://localhost:8000")
    parser.add_argument("--min-score",   type=float, default=float(os.getenv("CI_MIN_JUDGE_SCORE", "0.65")))
    args = parser.parse_args()

    if not OPENROUTER_API_KEY:
        print("[ERROR] OPENROUTER_API_KEY is not set")
        sys.exit(1)

    print(f"[CI EVAL] Gateway: {args.gateway_url}")
    print(f"[CI EVAL] Pass threshold: avg judge score >= {args.min_score}\n")

    score, evaluated = run_eval(args.gateway_url)

    print(f"\n[CI EVAL] Evaluated {evaluated}/{len(EVAL_QUERIES)} queries")
    print(f"[CI EVAL] Overall avg judge score: {score:.4f}")

    if evaluated == 0:
        print("❌ FAIL — no queries could be evaluated")
        sys.exit(1)

    if score >= args.min_score:
        print(f"✅ PASS — {score:.4f} >= {args.min_score}")
        sys.exit(0)
    else:
        print(f"❌ FAIL — {score:.4f} < {args.min_score}")
        sys.exit(1)


if __name__ == "__main__":
    main()
