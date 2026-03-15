from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from vectorizer import LocusVisualizer

app = FastAPI()
visualizer = LocusVisualizer()


@app.get("/")
def read_root():
    return {"status": "online", "service": "locus visual engine"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
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
    skip_rembg: str        = Form("false"),
    yolo_label: str        = Form(""),
    darken:     str        = Form("false"),
):
    image_data    = await file.read()
    has_yolo_box  = bool(yolo_label.strip())
    should_skip   = has_yolo_box or (skip_rembg.lower() == "true")
    should_darken = darken.lower() == "true"

    vector, category, confidence, debug_img = visualizer.process_image(
        image_bytes = image_data,
        skip_rembg  = should_skip,
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


# ── Text-guided category classification ───────────────────────────────────────
class TextClassifyRequest(BaseModel):
    title: str


@app.post("/classify-text")
async def classify_text(req: TextClassifyRequest):
    """
    Classifies a product title against the 15 canonical labels using
    CLIP's text encoder. Returns the best label and confidence, or
    null if the title doesn't match any fashion category.

    Used at indexing time to:
    1. Skip non-fashion products (bicycles, accessories, etc.)
    2. Guide vectorization so the pipeline knows what category to expect,
       helping it crop the right item from a model photo.
    """
    label, confidence = visualizer.classify_text(req.title)
    return {
        "title":      req.title,
        "category":   label,
        "confidence": round(confidence, 4),
    }