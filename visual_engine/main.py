from fastapi import FastAPI, UploadFile, File, Form
from vectorizer import LocusVisualizer

app = FastAPI()
visualizer = LocusVisualizer()

@app.get("/")
def read_root():
    return {"status": "online", "service": "locus visual engine"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_data = await file.read()
    
    # 1. run the detection pipeline
    detections, img_width, img_height = visualizer.detect_objects(image_data)
    
    # 2. return the results to the gateway
    return {
        "detections": detections,
        "image_width": img_width,
        "image_height": img_height
    }

@app.post("/vectorize")
async def vectorize(
    file: UploadFile = File(...),
    skip_rembg: str = Form("false"),
    yolo_label: str = Form(""),        # YOLO's label passed from gateway
):
    # 1. read the image correctly
    image_data = await file.read()
    
    # 2. check if yolo provided a label (meaning it detected a box)
    has_yolo_box = bool(yolo_label.strip())
    
    # 3. force skip_rembg to True if yolo found a box, otherwise rely on the form value
    should_skip_rembg = has_yolo_box or (skip_rembg.lower() == "true")
    
    # 4. pass ALL the arguments to your visualizer
    vector, category, confidence, debug_img = visualizer.process_image(
        image_bytes=image_data, 
        skip_rembg=should_skip_rembg, 
        yolo_label=yolo_label
    )
    
    # 5. handle errors if the visualizer failed (e.g., ghost image)
    if not vector:
        return {"error": "failed to process image"}
        
    # 6. return the correct dictionary to your gateway
    return {
        "filename": file.filename,
        "vector": vector, 
        "category": category, 
        "category_confidence": confidence,
        "debug_image": debug_img
    }