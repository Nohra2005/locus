# =============================================================================
# vectorizer.py
# Orchestrator — loads all models and coordinates the pipeline
#
# Detection models (run independently, results merged):
#   detector_clothing.py    — DeepFashion2 (shirts, pants, dresses, skirts...)
#   detector_accessories.py — YOLOv8 COCO (shoes, bags, ties...)
#
# Shared utilities:
#   CLIP  — vectorization + category classification
#   rembg — background removal
# =============================================================================

import torch
import io
import base64
import time
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from rembg import remove, new_session

from detector_clothing import ClothingDetector
from detector_accessories import AccessoryDetector


class LocusVisualizer:
    def __init__(self):

        # ── Detection Model 1 ─────────────────────────────────────────────────
        self.clothing_detector = ClothingDetector()

        # ── Detection Model 2 ─────────────────────────────────────────────────
        self.accessory_detector = AccessoryDetector()

        # ── CLIP ──────────────────────────────────────────────────────────────
        print("Loading CLIP (Vectorization & Classification)")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        # ── rembg ─────────────────────────────────────────────────────────────
        print("Loading rembg (Background Removal)")
        self.rembg_session = new_session("u2net")

        # ── CLIP labels ───────────────────────────────────────────────────────
        self.clip_labels = [
            "dress", "pants", "jeans", "shirt", "t-shirt",
            "jacket", "coat", "shoes", "sneakers", "bag",
            "handbag", "skirt", "shorts", "hat", "glasses", "watch"
        ]

        # Pre-compute text embeddings once at startup
        print("Pre-computing CLIP text embeddings...")
        text_inputs = self.clip_processor(
            text=self.clip_labels, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            self.text_features = self.clip_model.get_text_features(**text_inputs)
            self.text_features /= self.text_features.norm(p=2, dim=-1, keepdim=True)

        print("=" * 50)
        print("LOCUS VISUAL ENGINE READY")
        print("=" * 50)

    # =========================================================================
    # PUBLIC METHOD 1: detect_objects()
    # =========================================================================
    def detect_objects(self, image_bytes):
        """
        Runs both detection models independently on the same image.
        Merges their results into a single list for the user to pick from.

        Flow:
            1. ClothingDetector  → finds shirts, pants, dresses, etc.
            2. AccessoryDetector → finds shoes, bags, etc.
            3. Results merged (simple concatenation, no cross-model logic)
            4. Fallback to full-image CLIP if both models find nothing
        """
        t0 = time.time()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            W, H = image.size

            # Both detectors receive the same image and the same classify_fn.
            # They run completely independently.
            clothing    = self.clothing_detector.detect(image, self._classify_crop)
            accessories = self.accessory_detector.detect(image, self._classify_crop)

            # Merge — simple concatenation, no shared logic
            all_detections = clothing + accessories

            # Fallback if both models found nothing
            if not all_detections:
                print("Both models found nothing. Running full-image CLIP fallback.")
                clip_label, clip_conf = self._classify_crop(image)
                if clip_conf >= 0.35:
                    all_detections.append({
                        "bbox":         [0, 0, W, H],
                        "label":        clip_label,
                        "search_label": clip_label,
                        "score":        round(clip_conf, 3),
                        "source":       "clip_fallback"
                    })

            print(f"Total: {len(all_detections)} detections in {(time.time()-t0):.2f}s")
            return all_detections, W, H

        except Exception as e:
            print(f"detect_objects() error: {e}")
            return [], 0, 0

    # =========================================================================
    # PUBLIC METHOD 2: process_image()
    # =========================================================================
    def process_image(self, image_bytes):
        """
        Full pipeline for a single selected item:
        1. Remove background (rembg)
        2. Ghost image check
        3. Smart crop to content bounding box
        4. CLIP vectorization (512-dim vector for Qdrant)
        5. Category classification with 45% confidence threshold
        """
        t0 = time.time()
        try:
            try:
                input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
                original_size = input_image.size
            except Exception:
                print("Not a valid image file.")
                return None, None, None

            if max(input_image.size) > 512:
                input_image.thumbnail((512, 512))
                print(f"Resized {original_size} -> {input_image.size}")

            print("Removing background...")
            output_image = remove(input_image, session=self.rembg_session)

            alpha_max = output_image.getextrema()[3][1]
            if alpha_max == 0:
                print("Ghost image detected. Rejecting.")
                return None, None, None

            bbox = output_image.getbbox()
            if bbox:
                output_image = output_image.crop(bbox)

            white_bg = Image.new("RGB", output_image.size, (255, 255, 255))
            white_bg.paste(output_image, mask=output_image.split()[3])

            clip_inputs = self.clip_processor(images=white_bg, return_tensors="pt")
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**clip_inputs)

            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
            vector = image_features[0].tolist()

            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            top_score, top_idx = similarity[0].topk(1)
            confidence = top_score[0].item()
            best_label = self.clip_labels[top_idx[0]]

            if confidence < 0.45:
                print(f"Low confidence ({confidence:.2f}) for '{best_label}'. No category filter.")
                detected_category = None
            else:
                detected_category = best_label
                print(f"Category: {detected_category} ({confidence:.2f})")

            buf = io.BytesIO()
            white_bg.save(buf, format="PNG")
            debug_img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            print(f"process_image() done in {(time.time()-t0):.2f}s")
            return vector, detected_category, debug_img_b64

        except Exception as e:
            print(f"process_image() error: {e}")
            return None, None, None

    # =========================================================================
    # PRIVATE: _classify_crop()
    # Shared CLIP utility — passed into both detectors as classify_fn
    # =========================================================================
    def _classify_crop(self, pil_image):
        """
        Runs CLIP zero-shot classification on a PIL image.
        Returns (best_label, confidence) from self.clip_labels.
        """
        clip_inputs = self.clip_processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**clip_inputs)
        image_features /= image_features.norm(p=2, dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        top_score, top_idx = similarity[0].topk(1)

        return self.clip_labels[top_idx[0]], top_score[0].item()