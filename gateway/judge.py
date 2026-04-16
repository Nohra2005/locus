"""
Dual-provider async judge (Groq + Gemini) — fires per search, stores feedback with source="auto_judge".

Entry point: run_judge(query_image_bytes, results, gateway_base_url, groq_api_key, scores_out, gemini_api_key)
Called as a FastAPI BackgroundTask from the /search endpoint.
No imports from gateway.main — no circular dependencies.

Provider assignment: round-robin across available providers (result[i] → providers[i % n]).
If the assigned provider is in a rate-limit backoff window, the other provider is tried as fallback.
"""

from __future__ import annotations

import base64
import logging
import re
import threading
import time

import httpx

logger = logging.getLogger(__name__)

GROQ_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ── Per-provider rate limiter ──────────────────────────────────────────────────
# A single lock guards state for all providers so round-robin scheduling is
# coherent across concurrent background tasks.
#
# Rate targets (free tier):
#   Groq   — 6s gap  → ~10 RPM  (tight token-per-minute ceiling on vision models)
#   Gemini — 4s gap  → ~15 RPM  (gemini-2.0-flash free tier)

_provider_lock  = threading.Lock()
_provider_state: dict[str, dict] = {
    "groq":   {"last_call": 0.0, "backoff_until": 0.0, "min_gap": 6.0},
    "gemini": {"last_call": 0.0, "backoff_until": 0.0, "min_gap": 4.0},
}


def _acquire_slot(provider: str) -> bool:
    """
    Block until we're allowed to call the given provider.
    Returns False immediately if the provider is inside a rate-limit backoff window.
    """
    with _provider_lock:
        s   = _provider_state[provider]
        now = time.monotonic()
        if now < s["backoff_until"]:
            return False
        wait = s["last_call"] + s["min_gap"] - now
        if wait > 0:
            time.sleep(wait)
        s["last_call"] = time.monotonic()
        return True


def _set_backoff(provider: str, retry_after: float) -> None:
    """Record a rate-limit backoff window so all threads respect it."""
    with _provider_lock:
        _provider_state[provider]["backoff_until"] = time.monotonic() + retry_after
    logger.warning(f"judge: [{provider}] rate-limit backoff for {retry_after:.0f}s")


def _in_backoff(provider: str) -> bool:
    with _provider_lock:
        return time.monotonic() < _provider_state[provider]["backoff_until"]


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


# ── Provider implementations ───────────────────────────────────────────────────

def _judge_pair_groq(query_image_b64: str, result_image_url: str, groq_api_key: str) -> float | None:
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

    for attempt in range(3):
        if not _acquire_slot("groq"):
            logger.info("judge: [groq] skipping — inside backoff window")
            return None
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    GROQ_API_URL,
                    json=payload,
                    headers={"Authorization": f"Bearer {groq_api_key}"},
                )

            if resp.status_code == 429:
                retry_after_raw = (
                    resp.headers.get("retry-after")
                    or resp.headers.get("x-ratelimit-reset-requests")
                )
                try:
                    wait = float(retry_after_raw)
                except (TypeError, ValueError):
                    wait = 60.0
                _set_backoff("groq", wait)
                return None

            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            match = re.search(r"\d+\.\d+|\d+", content)
            if match is None:
                logger.warning(f"judge: [groq] could not parse score: {content!r}")
                return None
            return float(match.group())

        except httpx.HTTPStatusError as e:
            if attempt < 2:
                time.sleep(min(5.0 * (2 ** attempt), 30.0))
                continue
            try:
                body = e.response.json()
            except Exception:
                body = e.response.text
            logger.warning(f"judge: [groq] HTTP error {e.response.status_code} after 3 attempts: {body}")
            return None
        except Exception as e:
            logger.warning(f"judge: [groq] error: {e}")
            return None

    return None


def _judge_pair_gemini(query_image_b64: str, result_image_url: str, gemini_api_key: str) -> float | None:
    if not _acquire_slot("gemini"):
        logger.info("judge: [gemini] skipping — inside backoff window")
        return None

    result_b64 = fetch_image_as_base64(result_image_url)
    if result_b64 is None:
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=gemini_api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                JUDGE_PROMPT,
                types.Part.from_bytes(data=base64.b64decode(query_image_b64), mime_type="image/jpeg"),
                types.Part.from_bytes(data=base64.b64decode(result_b64),       mime_type="image/jpeg"),
            ],
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=10),
        )
        match = re.search(r"\d+\.\d+|\d+", response.text.strip())
        if match is None:
            logger.warning(f"judge: [gemini] could not parse score: {response.text!r}")
            return None
        return float(match.group())
    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower() or "resource_exhausted" in err.lower():
            _set_backoff("gemini", 60.0)
        else:
            logger.warning(f"judge: [gemini] error: {e}")
        return None


def judge_pair(query_image_b64: str, result_image_url: str, provider: str, api_key: str) -> float | None:
    if provider == "groq":
        return _judge_pair_groq(query_image_b64, result_image_url, api_key)
    elif provider == "gemini":
        return _judge_pair_gemini(query_image_b64, result_image_url, api_key)
    return None


# ── Entry point ────────────────────────────────────────────────────────────────

def run_judge(
    query_image_bytes: bytes,
    results: list[dict],
    gateway_base_url: str,
    groq_api_key: str,
    scores_out: dict | None = None,  # written into as each result is judged; enables frontend polling
    gemini_api_key: str = "",
) -> None:
    query_b64 = base64.b64encode(query_image_bytes).decode("utf-8")
    top3 = results[:3]

    # Build ordered provider list — only include keys that are set
    providers: list[tuple[str, str]] = []
    if groq_api_key:   providers.append(("groq",   groq_api_key))
    if gemini_api_key: providers.append(("gemini", gemini_api_key))
    if not providers:
        logger.warning("judge: no API keys configured, skipping")
        return

    logger.info(f"judge: starting — {len(top3)} result(s), providers={[p for p, _ in providers]}")

    for i, result in enumerate(top3):
        result_image_url = result.get("image_url", "")
        result_name      = result.get("name", "")
        product_id       = result.get("product_id", result_image_url)
        key              = product_id or result_image_url

        if not result_image_url:
            logger.info(f"judge: skipping '{result_name}' — no image_url")
            continue

        # Round-robin assignment with fallback if primary is rate-limited
        primary  = providers[i % len(providers)]
        fallback = providers[(i + 1) % len(providers)] if len(providers) > 1 else None

        provider, api_key = primary
        if _in_backoff(provider) and fallback:
            provider, api_key = fallback
            logger.info(f"judge: [{primary[0]}] in backoff — falling back to [{provider}]")

        logger.info(f"judge: [{provider}] scoring '{result_name}' ({result_image_url[:80]})")
        score = judge_pair(query_b64, result_image_url, provider, api_key)

        # If primary returned nothing, try the fallback before giving up
        if score is None and fallback:
            fallback_provider, fallback_key = fallback
            logger.info(f"judge: [{provider}] returned None — trying fallback [{fallback_provider}] for '{result_name}'")
            score = judge_pair(query_b64, result_image_url, fallback_provider, fallback_key)

        if score is None:
            logger.info(f"judge: no score for '{result_name}' from any provider")
            continue

        stars = score_to_stars(score)
        logger.info(f"judge: [{provider}] '{result_name}' → score={score:.2f} stars={stars} key={key[:60]}")

        # Write score immediately so the polling endpoint can return partial results
        if scores_out is not None and key:
            scores_out[key] = round(score, 3)

        feedback_payload = {
            "result_product_id": product_id,
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
