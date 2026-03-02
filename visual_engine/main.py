from fastapi import FastAPI, UploadFile, File, Form
from vectorizer import LocusVisualizer

app = FastAPI()
visualizer = LocusVisualizer()

@app.get("/")
def read_root():
    return {"status": "online", "service": "Locus Visual Engine"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_data = await file.read()
    detections, img_width, img_height = visualizer.detect_objects(image_data)
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
    # 1. Read the image correctly
    image_data = await file.read()
    
    # 2. Convert the string 'true'/'false' from the form to a Python boolean
    should_skip_rembg = (skip_rembg.lower() == "true")
    
    # 3. Pass ALL the arguments to your visualizer
    vector, category, confidence, debug_img = visualizer.process_image(
        image_bytes=image_data, 
        skip_rembg=should_skip_rembg, 
        yolo_label=yolo_label
    )
    
    # 4. Handle errors if the visualizer failed (e.g., ghost image)
    if not vector:
        return {"error": "Failed to process image"}
        
    # 5. Return the correct dictionary to your gateway
    return {
        "filename": file.filename,
        "vector": vector, 
        "category": category, 
        "category_confidence": confidence,
        "debug_image": debug_img
    }