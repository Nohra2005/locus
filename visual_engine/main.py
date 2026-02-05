from fastapi import FastAPI, UploadFile, File
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io
import rembg
import base64 # <--- Added for sending image back

app = FastAPI()

# Load Model (High-Res)
print("⏳ Loading AI Model (High-Res)...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
print("✅ High-Res Model Loaded!")

@app.post("/vectorize")
async def vectorize(file: UploadFile = File(...)):
    # 1. Read Image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # 2. Remove Background & Crop
    try:
        # Remove BG
        no_bg_image = rembg.remove(image)
        
        # Smart Crop (Zoom in on the item)
        bbox = no_bg_image.getbbox() 
        if bbox:
            no_bg_image = no_bg_image.crop(bbox)

        # Create white background for AI processing
        clean_image = Image.new("RGB", no_bg_image.size, (255, 255, 255))
        clean_image.paste(no_bg_image, mask=no_bg_image.split()[3])
        
    except Exception as e:
        print(f"⚠️ Preprocessing failed: {e}")
        clean_image = image

    # 3. Vectorize
    inputs = processor(images=clean_image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    vector = image_features[0].tolist()

    # 4. ENCODE IMAGE (The New Part)
    # Save the cut-out image to memory and convert to text
    buffered = io.BytesIO()
    clean_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {
        "vector": vector, 
        "processed_image": img_str  # <--- Sending this back!
    }