"""
Integration tests — requires the full stack running:
    docker compose up -d gateway visual_engine attribute_tagger mlflow

Run: pytest tests/test_pipeline.py -v
"""
import io
import json
import os
import time

import pytest
import requests
from PIL import Image

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")
VISUAL_URL  = os.getenv("VISUAL_URL",  "http://localhost:8001")
TAGGER_URL  = os.getenv("TAGGER_URL",  "http://localhost:8004")
TIMEOUT     = 30

_GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "mlops", "golden_images")
_PANTS_IMG  = os.path.join(_GOLDEN_DIR, "8cce3f151dbf3b2c5dd9.jpeg")   # pants
_DRESS_IMG  = os.path.join(_GOLDEN_DIR, "ae2d265877ca93ba9d26.jpeg")   # dress


@pytest.fixture(scope="module")
def pants_image():
    with open(_PANTS_IMG, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def dress_image():
    with open(_DRESS_IMG, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def grey_image():
    """Solid grey square — clearly not a fashion item."""
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color=(128, 128, 128)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def search_result(pants_image):
    """Shared search result used across multiple tests. Retries once if rate-limited."""
    for attempt in range(3):
        r = requests.post(
            f"{GATEWAY_URL}/search",
            files={"file": ("pants.jpg", io.BytesIO(pants_image), "image/jpeg")},
            timeout=TIMEOUT,
        )
        if r.status_code != 429:
            break
        time.sleep(62)
    r.raise_for_status()
    return r.json()


# ── Rate Limiting ──────────────────────────────────────────────────────────────
# Run first so we measure against a clean slate within the test minute window.

class TestRateLimiting:
    def test_search_rate_limit_enforced(self, pants_image):
        """10/min limit on /search — sending 15 requests should trigger a 429."""
        got_429 = False
        for _ in range(15):
            r = requests.post(
                f"{GATEWAY_URL}/search",
                files={"file": ("pants.jpg", io.BytesIO(pants_image), "image/jpeg")},
                timeout=TIMEOUT,
            )
            if r.status_code == 429:
                got_429 = True
                break
        assert got_429, "Expected a 429 after exceeding /search rate limit (10/min)"

    def test_detect_rate_limit_enforced(self, pants_image):
        """5/min limit on /detect — sending 10 requests with a real image should trigger a 429."""
        got_429 = False
        for _ in range(10):
            r = requests.post(
                f"{GATEWAY_URL}/detect",
                files={"file": ("pants.jpg", io.BytesIO(pants_image), "image/jpeg")},
                timeout=TIMEOUT,
            )
            if r.status_code == 429:
                got_429 = True
                break
        assert got_429, "Expected a 429 after exceeding /detect rate limit (5/min)"


# ── Search ─────────────────────────────────────────────────────────────────────

class TestSearch:
    def test_returns_matches(self, search_result):
        assert "matches" in search_result
        assert len(search_result["matches"]) > 0

    def test_response_schema(self, search_result):
        assert "search_id" in search_result
        assert "detected_category" in search_result
        assert "matches" in search_result
        match = search_result["matches"][0]
        assert "product_id" in match
        assert "score" in match
        assert isinstance(match["score"], float)
        assert 0.0 <= match["score"] <= 1.0

    def test_results_sorted_by_score(self, search_result):
        scores = [m["score"] for m in search_result["matches"]]
        assert scores == sorted(scores, reverse=True), "Results should be sorted by score descending"

    def test_detects_clothing_category(self, search_result):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "visual_engine"))
        from clip_labels import CANONICAL_LABELS
        assert search_result.get("detected_category") in CANONICAL_LABELS

    def test_dress_detected_as_dress(self, dress_image):
        r = requests.post(
            f"{GATEWAY_URL}/search",
            files={"file": ("dress.jpg", io.BytesIO(dress_image), "image/jpeg")},
            timeout=TIMEOUT,
        )
        assert r.status_code == 200
        assert r.json().get("detected_category") == "dress"

    def test_missing_file_returns_422(self):
        r = requests.post(f"{GATEWAY_URL}/search", timeout=TIMEOUT)
        assert r.status_code == 422

    def test_corrupt_image_does_not_crash(self):
        r = requests.post(
            f"{GATEWAY_URL}/search",
            files={"file": ("bad.jpg", io.BytesIO(b"not an image at all"), "image/jpeg")},
            timeout=TIMEOUT,
        )
        assert r.status_code in (200, 400, 422), (
            f"Corrupt image caused unexpected status: {r.status_code}"
        )

    def test_non_fashion_image_handled(self, grey_image):
        r = requests.post(
            f"{GATEWAY_URL}/search",
            files={"file": ("grey.jpg", io.BytesIO(grey_image), "image/jpeg")},
            timeout=TIMEOUT,
        )
        assert r.status_code == 200, "Non-fashion image should not crash the endpoint"


# ── Detect ─────────────────────────────────────────────────────────────────────

class TestDetect:
    def test_returns_list(self, pants_image):
        r = requests.post(
            f"{GATEWAY_URL}/detect",
            files={"file": ("pants.jpg", io.BytesIO(pants_image), "image/jpeg")},
            timeout=TIMEOUT,
        )
        assert r.status_code == 200
        data = r.json()
        detections = data["detections"] if isinstance(data, dict) else data
        assert isinstance(detections, list)

    def test_detection_has_bounding_box(self, pants_image):
        r = requests.post(
            f"{GATEWAY_URL}/detect",
            files={"file": ("pants.jpg", io.BytesIO(pants_image), "image/jpeg")},
            timeout=TIMEOUT,
        )
        data = r.json()
        detections = data["detections"] if isinstance(data, dict) else data
        if detections:
            d = detections[0]
            assert "bbox" in d or "box" in d or all(k in d for k in ("x1", "y1", "x2", "y2")), (
                f"Detection missing bounding box: {d}"
            )

    def test_corrupt_image_does_not_crash(self):
        r = requests.post(
            f"{GATEWAY_URL}/detect",
            files={"file": ("bad.jpg", io.BytesIO(b"not an image"), "image/jpeg")},
            timeout=TIMEOUT,
        )
        assert r.status_code in (200, 400)


# ── Feedback ───────────────────────────────────────────────────────────────────

class TestFeedback:
    def test_submit_feedback(self, search_result):
        product_id = search_result["matches"][0]["product_id"]

        r = requests.post(
            f"{GATEWAY_URL}/feedback",
            json={
                "result_product_id": product_id,
                "rating":            5,
                "source":            "test",
            },
            timeout=TIMEOUT,
        )
        assert r.status_code == 200

    def test_get_feedback_returns_list(self):
        r = requests.get(f"{GATEWAY_URL}/feedback", timeout=TIMEOUT)
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, (list, dict))

    def test_low_rating_accepted(self, search_result):
        product_id = search_result["matches"][0]["product_id"]
        r = requests.post(
            f"{GATEWAY_URL}/feedback",
            json={"result_product_id": product_id, "rating": 1, "source": "test"},
            timeout=TIMEOUT,
        )
        assert r.status_code == 200


# ── CORS ───────────────────────────────────────────────────────────────────────

class TestCORS:
    def test_cors_header_present_on_options(self):
        r = requests.options(
            f"{GATEWAY_URL}/search",
            headers={
                "Origin":                        "http://example.com",
                "Access-Control-Request-Method": "POST",
            },
            timeout=TIMEOUT,
        )
        headers_lower = {k.lower(): v for k, v in r.headers.items()}
        assert "access-control-allow-origin" in headers_lower, (
            f"CORS header missing. Response headers: {dict(r.headers)}"
        )

    def test_cors_header_on_get(self):
        r = requests.get(
            f"{GATEWAY_URL}/",
            headers={"Origin": "http://example.com"},
            timeout=TIMEOUT,
        )
        headers_lower = {k.lower(): v for k, v in r.headers.items()}
        assert "access-control-allow-origin" in headers_lower


# ── Attribute Tagger ───────────────────────────────────────────────────────────

class TestAttributeTagger:
    def test_tagger_health(self):
        r = requests.get(f"{TAGGER_URL}/health", timeout=TIMEOUT)
        assert r.status_code == 200

    def test_attributes_populated_after_search(self, pants_image):
        for attempt in range(3):
            r = requests.post(
                f"{GATEWAY_URL}/search",
                files={"file": ("pants.jpg", io.BytesIO(pants_image), "image/jpeg")},
                timeout=TIMEOUT,
            )
            if r.status_code != 429:
                break
            time.sleep(62)
        assert r.status_code == 200, f"Search failed with {r.status_code}"
        search_id = r.json().get("search_id")
        assert search_id

        # Background task — give it time to complete
        time.sleep(5)

        attr = requests.get(f"{GATEWAY_URL}/search/{search_id}/attributes", timeout=TIMEOUT)
        assert attr.status_code == 200


# ── Index & Health ─────────────────────────────────────────────────────────────

class TestIndexAndHealth:
    def test_gateway_health(self):
        r = requests.get(f"{GATEWAY_URL}/", timeout=TIMEOUT)
        assert r.status_code == 200

    def test_visual_engine_health(self):
        r = requests.get(f"{VISUAL_URL}/", timeout=TIMEOUT)
        assert r.status_code == 200

    def test_index_stats_returns_count(self):
        r = requests.get(f"{GATEWAY_URL}/index-stats", timeout=TIMEOUT)
        assert r.status_code == 200
        assert isinstance(r.json(), dict)

    def test_index_is_not_empty(self):
        r = requests.get(f"{GATEWAY_URL}/index-stats", timeout=TIMEOUT)
        data = r.json()
        counts = [v for v in data.values() if isinstance(v, (int, float))]
        assert any(c > 0 for c in counts), f"Index appears empty: {data}"
