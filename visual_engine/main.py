# LOCUS: main.py
from fastapi import FastAPI, UploadFile, File
from vectorizer import LocusVisualizer
import shutil
import os

app = FastAPI()

# Initialize the AI model once when the app starts
visualizer = LocusVisualizer()

@app.get("/")
def read_root():
    return {"status": "online", "service": "Locus Visual Engine"}

@app.post("/vectorize")
async def vectorize_image(file: UploadFile = File(...)):
    # 1. Save the uploaded file temporarily
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Convert image to vector
    vector = visualizer.get_vector(temp_filename)

    # 3. Clean up (delete temp file)
    os.remove(temp_filename)

    if vector:
        return {"filename": file.filename, "vector": vector}
    else:
        return {"error": "Failed to process image"}