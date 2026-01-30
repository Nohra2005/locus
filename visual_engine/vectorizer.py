# LOCUS: vectorizer.py
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from rembg import remove
import io

class LocusVisualizer:
    def __init__(self):
        print("⏳ Loading CLIP and RemBG models...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("✅ Models loaded successfully!")

    def get_vector(self, image_path):
        """
        Removes background and then generates a 512-dim vector.
        """
        try:
            # 1. Open the image
            input_image = Image.open(image_path).convert("RGBA")

            # 2. THE FIX: Strip the background (the noisy street, cars, etc.)
            # This makes the non-dress areas transparent
            output_image = remove(input_image)

            # 3. Create a white background (CLIP performs better on white than transparency)
            white_bg = Image.new("RGB", output_image.size, (255, 255, 255))
            white_bg.paste(output_image, mask=output_image.split()[3]) 

            # 4. Generate the vector from the CLEANED image
            inputs = self.processor(images=white_bg, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)

            return outputs[0].tolist()
            
        except Exception as e:
            print(f"❌ Error during preprocessing/vectorization: {e}")
            return None