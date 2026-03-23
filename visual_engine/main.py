import json
import os

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from vectorizer import LocusVisualizer

app = FastAPI()
visualizer = LocusVisualizer()

OVERRIDES_PATH = "/app/whitelist_overrides.json"


@app.get("/")
def read_root():
    return {"status": "online", "service": "locus visual engine"}


# ── Search path ────────────────────────────────────────────────────────────────

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """Search time. Returns all YOLO boxes for user to select from."""
    image_data = await file.read()
    detections, img_width, img_height = visualizer.detect_objects(image_data)
    return {
        "detections":   detections,
        "image_width":  img_width,
        "image_height": img_height,
    }


@app.post("/vectorize")
async def vectorize(
    file:       UploadFile = File(...),
    yolo_label: str        = Form(""),
    darken:     str        = Form("false"),
):
    """
    Search time. Expects ALREADY CROPPED image bytes — gateway crops before calling.
    yolo_label: canonical category from user's selected box.
    """
    image_data    = await file.read()
    should_darken = darken.lower() == "true"

    vector, category, confidence, debug_img = visualizer.process_image(
        image_bytes = image_data,
        yolo_label  = yolo_label,
        darken      = should_darken,
    )

    if not vector:
        return {"error": "failed to process image"}

    return {
        "filename":            file.filename,
        "vector":              vector,
        "category":            category,
        "category_confidence": confidence,
        "debug_image":         debug_img,
    }


# ── Index path ─────────────────────────────────────────────────────────────────

@app.post("/index-image")
async def index_image(
    file:  UploadFile = File(...),
    title: str        = Form(""),
):
    """
    Index time only. One HTTP call per product does everything:
      _classify_title() → YOLO detect → 3-tier crop → CLIP embed
    Returns vector + category + box_source, or {"skipped": true} with reason.
    """
    image_data = await file.read()
    result     = visualizer.index_product(image_data, title=title)
    return result


# ── Debug index (dashboard only — does NOT write to Qdrant) ───────────────────

@app.post("/debug-index")
async def debug_index(
    file:  UploadFile = File(...),
    title: str        = Form(""),
):
    """
    Same logic as /index-image but for dashboard preview only.
    Does NOT write to Qdrant.
    """
    image_data = await file.read()
    result     = visualizer.index_product(image_data, title=title)
    return result


# ── Classify text (debug) ──────────────────────────────────────────────────────

class ClassifyTextRequest(BaseModel):
    title: str


@app.post("/classify-text")
async def classify_text(req: ClassifyTextRequest):
    category, confidence = visualizer.classify_text(req.title)
    return {"category": category, "confidence": confidence}


# ══════════════════════════════════════════════════════════════════════════════
# WHITELIST OVERRIDE ENDPOINTS
#
# These allow the gateway to add approved whitelist suggestions to the
# visual engine's runtime token map without restarting the container.
#
# Flow:
#   1. Shop owner suggests word in dashboard → gateway writes to pending_whitelist.json
#   2. Developer approves in dashboard → gateway calls POST /whitelist-add
#   3. Visual engine appends to whitelist_overrides.json + reloads token map
#   4. Gateway triggers re-index of matching skipped products
# ══════════════════════════════════════════════════════════════════════════════

class WhitelistAddRequest(BaseModel):
    word:     str
    category: str


def _read_overrides() -> list:
    if not os.path.exists(OVERRIDES_PATH):
        return []
    try:
        with open(OVERRIDES_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _write_overrides(data: list):
    with open(OVERRIDES_PATH, "w") as f:
        json.dump(data, f, indent=2)


@app.post("/whitelist-add")
async def whitelist_add(req: WhitelistAddRequest):
    """
    Called by gateway when a whitelist suggestion is approved.
    Appends the word → category mapping to whitelist_overrides.json,
    then reloads the in-memory token map immediately.
    """
    word     = req.word.strip().lower()
    category = req.category.strip()

    if not word or not category:
        return {"error": "word and category are required"}

    overrides = _read_overrides()

    # Update existing entry if word already present, otherwise append
    existing = next((e for e in overrides if e.get("word") == word), None)
    if existing:
        existing["category"] = category
        existing["status"]   = "approved"
    else:
        overrides.append({
            "word":     word,
            "category": category,
            "status":   "approved",
        })

    _write_overrides(overrides)

    # Reload the in-memory token map immediately
    result = visualizer.reload_overrides()

    print(f"[WHITELIST] Added override: '{word}' → '{category}'")
    return {
        "status":   "added",
        "word":     word,
        "category": category,
        "reload":   result,
    }


@app.post("/whitelist-reload")
async def whitelist_reload():
    """
    Force reload the token map from whitelist_overrides.json.
    Call this if you manually edited the file.
    """
    result = visualizer.reload_overrides()
    return {"status": "reloaded", **result}


@app.get("/whitelist-overrides")
async def whitelist_overrides():
    """
    List all currently active override entries.
    """
    overrides = _read_overrides()
    active    = [e for e in overrides if e.get("status") == "approved"]
    return {
        "total":   len(active),
        "entries": active,
    }