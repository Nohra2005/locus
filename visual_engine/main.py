from fastapi import FastAPI, UploadFile, File
from vectorizer import LocusVisualizer

app = FastAPI()

# Initialize the logic class once
visualizer = LocusVisualizer()

@app.get("/")
def read_root():
    return {"status": "online", "service": "Locus Visual Engine"}

@app.post("/vectorize")
async def vectorize(file: UploadFile = File(...)):
    # 1. Read Bytes
    image_data = await file.read()
    
    # 2. Delegate to Vectorizer
    vector, category, debug_image = visualizer.process_image(image_data)

    if vector:
        return {
            "filename": file.filename, 
            "vector": vector,
            "category": category,         # Your Feature
            "processed_image": debug_image # Partner's Feature
        }
    else:
        return {"error": "Failed to process image"}