"""
Attribute tagger — calls Gemini 2.0 Flash (VLM) via OpenRouter to extract
rich visual attributes from a cropped clothing image.

Returns a structured dict: colors, style, silhouette, occasion, pattern,
material_feel, trend_tags. Returns {} on any failure — never raises.
"""

from __future__ import annotations

import json
import logging
import os
import re

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_MODEL   = "google/gemini-2.0-flash-001"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

TAGGER_PROMPT = """\
You are a fashion attribute extraction system. Analyze the clothing item in this image.

Return ONLY a valid JSON object with exactly these keys:
{
  "colors": ["list", "of", "1-3", "dominant colors"],
  "style": "one style label",
  "silhouette": "brief shape/cut description",
  "occasion": ["list", "of", "1-2", "occasions"],
  "pattern": "one pattern label",
  "material_feel": "one material impression",
  "trend_tags": ["list", "of", "0-2", "trend tags"]
}

Guidelines:
- colors: e.g. "navy blue", "ivory", "forest green", "hot pink", "camel"
- style: one of casual, formal, streetwear, athletic, bohemian, minimalist, preppy, \
romantic, quiet luxury, edgy, classic, resort
- silhouette: e.g. "fitted blazer", "flowy midi dress", "slim-fit jeans", \
"oversized hoodie", "a-line skirt"
- occasion: e.g. "office", "casual", "evening", "beach", "workout", "date night", \
"weekend", "formal event"
- pattern: one of solid, striped, floral, plaid, animal print, geometric, \
abstract, polka dot, tie-dye, logo print
- material_feel: one of cotton, silk, denim, knitwear, leather, chiffon, \
linen, velvet, satin, wool, synthetic
- trend_tags: e.g. "old money", "Y2K", "cottagecore", "quiet luxury", "dark academia", \
"coastal grandmother", "clean girl", "gorpcore"

Respond ONLY with valid JSON. No markdown. No explanation. No extra text."""


def tag_image(image_b64: str, category: str) -> dict:
    """
    Calls Gemini 2.0 Flash with the cropped image and returns a structured
    attribute dict. Returns {} on any failure.
    """
    if not OPENROUTER_API_KEY:
        logger.warning("tagger: OPENROUTER_API_KEY not set — skipping")
        return {}

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": TAGGER_PROMPT},
                ],
            }
        ],
        "max_tokens": 300,
        "temperature": 0.0,
    }

    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.post(
                OPENROUTER_API_URL,
                json=payload,
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

        # Strip markdown code fences if Gemini added them despite instructions
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        attrs = json.loads(content)
        logger.info(f"tagger: extracted attributes for category={category}: {attrs}")
        return attrs

    except json.JSONDecodeError as e:
        logger.warning(f"tagger: JSON parse failed: {e} — raw: {content!r}")
        return {}
    except Exception as e:
        logger.warning(f"tagger: request failed: {e}")
        return {}
