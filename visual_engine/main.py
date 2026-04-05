import io
import base64
import json
import os

from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image, ImageDraw
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from vectorizer import LocusVisualizer

app = FastAPI()
Instrumentator().instrument(app).expose(app)
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
    Index time only. Returns vector + category + box_source,
    or {"skipped": true} with reason.
    """
    image_data = await file.read()
    result     = visualizer.index_product(image_data, title=title)
    return result


# ── Debug index (dashboard preview — does NOT write to Qdrant) ────────────────

@app.post("/debug-index")
async def debug_index(
    file:  UploadFile = File(...),
    title: str        = Form(""),
):
    """
    Same logic as /index-image but returns an annotated image showing:
      - All YOLO detections as grey dashed boxes with label + score
      - The selected crop box in gold (successfully classified)
      - Red border only if skipped

    Does NOT write to Qdrant.
    """
    image_data = await file.read()
    result     = visualizer.index_product(image_data, title=title)

    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image.thumbnail((600, 600))
        scale_x = image.width  / Image.open(io.BytesIO(image_data)).width
        scale_y = image.height / Image.open(io.BytesIO(image_data)).height
        W, H    = image.size
        draw    = ImageDraw.Draw(image)

        all_detections = result.get("all_detections", [])
        selected_bbox  = result.get("selected_bbox", None)

        # Draw all non-selected boxes in grey
        for det in all_detections:
            bx1, by1, bx2, by2 = det["bbox"]
            bx1 = int(bx1 * scale_x); by1 = int(by1 * scale_y)
            bx2 = int(bx2 * scale_x); by2 = int(by2 * scale_y)

            # Skip if this is the selected box — we'll draw it in gold below
            if selected_bbox:
                sx1, sy1, sx2, sy2 = [
                    int(selected_bbox[0] * scale_x),
                    int(selected_bbox[1] * scale_y),
                    int(selected_bbox[2] * scale_x),
                    int(selected_bbox[3] * scale_y),
                ]
                if abs(bx1 - sx1) < 5 and abs(by1 - sy1) < 5:
                    continue

            draw.rectangle([bx1, by1, bx2, by2], outline="#555555", width=2)
            lbl = f"{det.get('search_label','')} {det.get('score', 0):.2f}"
            draw.rectangle([bx1, by1 - 16, bx1 + len(lbl) * 6, by1], fill="#333333")
            draw.text((bx1 + 2, by1 - 14), lbl, fill="#aaaaaa")

        if result.get("skipped"):
            # Red border for skipped products
            for i in range(4):
                draw.rectangle([i, i, W - i, H - i], outline="#c97070")
            draw.rectangle([0, H - 28, W, H], fill="#c97070")
            draw.text((6, H - 21), result.get("skip_reason", "skipped"), fill="#ffffff")

        elif selected_bbox:
            # Gold box for the selected crop
            sx1 = int(selected_bbox[0] * scale_x)
            sy1 = int(selected_bbox[1] * scale_y)
            sx2 = int(selected_bbox[2] * scale_x)
            sy2 = int(selected_bbox[3] * scale_y)

            for i in range(3):
                draw.rectangle([sx1 + i, sy1 + i, sx2 - i, sy2 - i], outline="#c9a96e")

            # Gold label showing category + box_source
            cat     = result.get("category", "")
            box_src = result.get("box_source", "")
            lbl     = f"✓ {cat}  [{box_src}]"
            draw.rectangle([sx1, sy2, sx1 + len(lbl) * 7, sy2 + 18], fill="#c9a96e")
            draw.text((sx1 + 3, sy2 + 3), lbl, fill="#000000")

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        result["debug_image"] = base64.b64encode(buf.getvalue()).decode("utf-8")

    except Exception as e:
        print(f"[DEBUG-INDEX] Image annotation failed: {e}")
        result["debug_image"] = None

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
    word     = req.word.strip().lower()
    category = req.category.strip()

    if not word or not category:
        return {"error": "word and category are required"}

    overrides = _read_overrides()
    existing  = next((e for e in overrides if e.get("word") == word), None)
    if existing:
        existing["category"] = category
        existing["status"]   = "approved"
    else:
        overrides.append({"word": word, "category": category, "status": "approved"})

    _write_overrides(overrides)
    result = visualizer.reload_overrides()

    print(f"[WHITELIST] Added override: '{word}' → '{category}'")
    return {"status": "added", "word": word, "category": category, "reload": result}


@app.post("/whitelist-reload")
async def whitelist_reload():
    result = visualizer.reload_overrides()
    return {"status": "reloaded", **result}


@app.get("/whitelist-overrides")
async def whitelist_overrides():
    overrides = _read_overrides()
    active    = [e for e in overrides if e.get("status") == "approved"]
    return {"total": len(active), "entries": active}