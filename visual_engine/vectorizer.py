# LOCUS: vectorizer.py
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class LocusVisualizer:
    def __init__(self):
        print("⏳ Loading CLIP model... (this may take a moment)")
        # Load the pre-trained model and processor
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        print("✅ CLIP model loaded successfully!")

    def get_vector(self, image_path):
        """
        Takes an image file path, processes it, and returns a vector (list of floats).
        """
        try:
            # 1. Open the image
            image = Image.open(image_path)

            # 2. Process the image (resize, normalize) for the model
            inputs = self.processor(images=image, return_tensors="pt")

            # 3. Generate the "embedding" (the vector)
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)

            # 4. Convert to a simple list of numbers
            # outputs[0] grabs the first (and only) vector
            vector = outputs[0].tolist()
            
            return vector
            
        except Exception as e:
            print(f"❌ Error vectorizing image: {e}")
            return None

# Simple test to run if this file is executed directly
if __name__ == "__main__":
    # Create a dummy image just to test the model
    img = Image.new('RGB', (100, 100), color = 'red')
    img.save("test_image.jpg")
    
    viz = LocusVisualizer()
    vector = viz.get_vector("test_image.jpg")
    
    print(f"Vector generated with length: {len(vector)}")
    print(f"First 5 numbers: {vector[:5]}")