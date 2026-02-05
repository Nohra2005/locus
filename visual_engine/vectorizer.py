import torch
import io
import base64
import time
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from rembg import remove, new_session

class LocusVisualizer:
    def __init__(self):
        print("⏳ Loading CLIP and RemBG models...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        
        # Pre-load session to avoid first-run lag
        print("🔧 Pre-loading Background Remover...")
        self.rembg_session = new_session("u2net") 
        
        # Categories
        self.labels = [
            "dress", "pants", "jeans", "shirt", "t-shirt", 
            "jacket", "coat", "shoes", "sneakers", "bag", 
            "handbag", "skirt", "shorts", "hat", "glasses", "watch"
        ]
        
        print("🧠 Memorizing categories...")
        inputs = self.processor(text=self.labels, return_tensors="pt", padding=True)
        with torch.no_grad():
            self.text_features = self.model.get_text_features(**inputs)
            self.text_features /= self.text_features.norm(p=2, dim=-1, keepdim=True)
            
        print("✅ Locus Visual Engine Ready!")

    def process_image(self, image_bytes):
        t0 = time.time()
        try:
            # --- MITIGATION 1: Input Validation ---
            try:
                input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
                original_size = input_image.size
            except Exception:
                print("❌ Edge Case: File is not a valid image.")
                return None, None, None

            # --- SPEED FIX: Resize BEFORE processing ---
            if max(input_image.size) > 512:
                input_image.thumbnail((512, 512))
                print(f"⚡ [SPEED FIX] Resized from {original_size} to {input_image.size}")
            else:
                print(f"ℹ️ Image is small ({original_size}), skipping resize.")

            # Remove Background
            print("✂️ Removing background...")
            output_image = remove(input_image, session=self.rembg_session)

            # --- MITIGATION 2: The "Ghost Image" Check ---
            # If the image is completely transparent (rembg removed everything), reject it.
            # This prevents sending empty pixels to the AI.
            extrema = output_image.getextrema()
            alpha_extrema = extrema[3] # (min_alpha, max_alpha)
            if alpha_extrema[1] == 0:
                print("❌ Edge Case: No object detected (Image is empty after RemBG).")
                return None, None, None

            # Smart Crop & White BG (Partner's Logic)
            bbox = output_image.getbbox()
            if bbox:
                output_image = output_image.crop(bbox)

            white_bg = Image.new("RGB", output_image.size, (255, 255, 255))
            white_bg.paste(output_image, mask=output_image.split()[3]) 

            # Vectorize
            inputs = self.processor(images=white_bg, return_tensors="pt")
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
            vector = image_features[0].tolist()
            
            # --- MITIGATION 3: Confidence Threshold (The "Cat" Fix) ---
            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(1)
            
            confidence_score = values[0].item()
            best_label = self.labels[indices[0]]

            # STRICT THRESHOLD: 35%
            # Rejects inputs that don't look like *any* of our categories
            if confidence_score < 0.45: 
                print(f"⚠️ Edge Case: Ambiguous Object. Best guess '{best_label}' was only {confidence_score:.2f} confident.")
                detected_category = None 
            else:
                detected_category = best_label
                print(f"👁️ Detected: {detected_category} ({confidence_score:.2f})")

            # Encode Debug Image
            buffered = io.BytesIO()
            white_bg.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            print(f"🚀 Finished in {(time.time()-t0):.2f} seconds")
            return vector, detected_category, img_str

        except Exception as e:
            print(f"❌ Error: {e}")
            return None, None, None