"""
Groq live async judge — fires per search, stores feedback with source="auto_judge".

Entry point: run_judge(query_image_bytes, results, gateway_base_url, groq_api_key)
Called as a FastAPI BackgroundTask from the /search endpoint.
No imports from gateway.main — no circular dependencies.
"""

from __future__ import annotations

import base64
import logging
import re

import httpx

logger = logging.getLogger(__name__)

GROQ_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
JUDGE_PROMPT = (
    "You are a fashion visual similarity expert.\n"
    "You will be shown two clothing images: first the QUERY image, then a RESULT image.\n"
    "Rate how visually similar the result is to the query.\n"
    "Give a score from 0.00 to 1.00 using exactly two decimal places.\n"
    "Anchor points:\n"
    "  1.00 = identical item\n"
    "  0.80 = very similar (same style, colour, silhouette)\n"
    "  0.60 = similar style (same category, close design)\n"
    "  0.40 = same category only\n"
    "  0.20 = loosely related\n"
    "  0.00 = unrelated\n"
    "Respond with ONLY the numeric score. No explanation."
)


def score_to_stars(score: float) -> int:
    if score >= 0.80:
        return 5
    elif score >= 0.60:
        return 4
    elif score >= 0.40:
        return 3
    elif score >= 0.20:
        return 2
    else:
        return 1


def fetch_image_as_base64(url: str) -> str | None:
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return base64.b64encode(resp.content).decode("utf-8")
    except Exception as e:
        logger.warning(f"judge: failed to fetch image {url}: {e}")
        return None


def judge_pair(query_image_b64: str, result_image_url: str, groq_api_key: str) -> float | None:
    result_b64 = fetch_image_as_base64(result_image_url)
    if result_b64 is None:
        return None

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": JUDGE_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{query_image_b64}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{result_b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 10,
        "temperature": 0.0,
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                GROQ_API_URL,
                json=payload,
                headers={"Authorization": f"Bearer {groq_api_key}"},
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            match = re.search(r"\d+\.\d+|\d+", content)
            if match is None:
                logger.warning(f"judge: could not parse score from response: {content!r}")
                return None
            return float(match.group())
    except httpx.HTTPStatusError as e:
        try:
            body = e.response.json()
        except Exception:
            body = e.response.text
        logger.warning(f"judge: Groq API error {e.response.status_code}: {body}")
        return None
    except Exception as e:
        logger.warning(f"judge: Groq API error: {e}")
        return None


def run_judge(
    query_image_bytes: bytes,
    results: list[dict],
    gateway_base_url: str,
    groq_api_key: str,
) -> None:
    query_b64 = base64.b64encode(query_image_bytes).decode("utf-8")
    top5 = results[:5]

    for result in top5:
        result_image_url = result.get("image_url", "")
        result_name      = result.get("name", "")

        if not result_image_url:
            continue

        score = judge_pair(query_b64, result_image_url, groq_api_key)
        if score is None:
            continue

        stars = score_to_stars(score)
        logger.debug(f"judge: {result_name} → score={score:.2f} stars={stars}")

        feedback_payload = {
            "result_product_id": result.get("product_id", ""),
            "result_image_url":  result_image_url,
            "result_name":       result_name,
            "store_name":        result.get("store_name", ""),
            "category":          result.get("category_tag", ""),
            "rating":            stars,
            "source":            "auto_judge",
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(f"{gateway_base_url}/feedback", json=feedback_payload)
                resp.raise_for_status()
        except Exception as e:
            logger.warning(f"judge: failed to post feedback for '{result_name}': {e}")
