from fastapi import FastAPI, UploadFile, File
from vectorizer import LocusVisualizer

app = FastAPI()

# Initialize the logic class once
visualizer = LocusVisualizer()

@app.get("/")
def read_root():
    return {"status": "online", "service": "Locus Visual Engine"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    NEW ENDPOINT: Detects all fashion objects in an image.
    Returns a list of bounding boxes with labels.
    The user will then pick which one to search for.
    """
    image_data = await file.read()
    detections, img_width, img_height = visualizer.detect_objects(image_data)
    
    return {
        "detections": detections,
        "image_width": img_width,
        "image_height": img_height
    }

@app.post("/vectorize")
async def vectorize(file: UploadFile = File(...)):
    """
    Existing endpoint: vectorizes a single (pre-cropped) image.
    Called AFTER the user selects an object from /detect results.
    """
    image_data = await file.read()
    vector, category, debug_image = visualizer.process_image(image_data)

    if vector:
        return {
            "filename": file.filename, 
            "vector": vector,
            "category": category,
            "processed_image": debug_image
        }
    else:
        return {"error": "Failed to process image"}