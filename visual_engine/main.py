from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from vectorizer import LocusVisualizer

app = FastAPI()
visualizer = LocusVisualizer()


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
      _classify_title() → YOLO detect → crop → CLIP × 2

    Category is ALWAYS determined by title classifier — YOLO is geometry only.
    Returns both vectors + category, or {"skipped": true} with reason.
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
    Same logic as /index-image but returns an annotated image showing:
      - All YOLO boxes in grey
      - The selected crop box highlighted in gold
      - Category decision + signals
    Does NOT write to Qdrant. Used by the dev dashboard for box preview.
    """
    import io as _io
    import base64
    from PIL import Image as PILImage, ImageDraw

    image_data = await file.read()

    # Run indexing logic (without saving)
    result = visualizer.index_product(image_data, title=title)

    # Run detection to get all boxes for visualization
    detections, W, H = visualizer.detect_objects(image_data)

    # Draw annotated image
    image = PILImage.open(_io.BytesIO(image_data)).convert("RGB")
    draw  = ImageDraw.Draw(image)

    # All boxes in grey
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="#555555", width=2)
        lbl = f"{det.get('search_label','?')} {det['score']:.2f}"
        draw.rectangle([x1, y1 - 14, x1 + len(lbl) * 6, y1], fill="#555555")
        draw.text((x1 + 2, y1 - 13), lbl, fill="#ffffff")

    # Highlight the selected box in gold
    if not result.get("skipped") and result.get("box_source") != "full_image":
        final_cat = result.get("category", "")
        matching  = [d for d in detections if d.get("search_label") == final_cat]
        if matching:
            best       = max(matching, key=lambda d: d["score"])
            x1, y1, x2, y2 = best["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline="#c9a96e", width=4)
            lbl = f"✓ {final_cat}"
            draw.rectangle([x1, y1 - 18, x1 + len(lbl) * 8, y1], fill="#c9a96e")
            draw.text((x1 + 2, y1 - 16), lbl, fill="#000000")

    buf = _io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    debug_img = base64.b64encode(buf.getvalue()).decode()

    return {
        **result,
        "debug_image":  debug_img,
        "all_boxes":    detections,
        "image_width":  W,
        "image_height": H,
    }


# ── Text classification (debug utility) ───────────────────────────────────────

class TextClassifyRequest(BaseModel):
    title: str


@app.post("/classify-text")
async def classify_text(req: TextClassifyRequest):
    """Debug endpoint — classify a product title using the 3-layer cascade."""
    label, confidence = visualizer.classify_text(req.title)
    return {
        "title":      req.title,
        "category":   label,
        "confidence": round(confidence, 4),
    }