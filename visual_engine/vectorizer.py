import torch
import io
import base64
import time
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from rembg import remove, new_session
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# ─── DeepFashion2 class labels ────────────────────────────────────────────────
# These are the 13 clothing categories the model was trained on.
# Unlike generic YOLO (which only knows "person", "handbag"...), this model
# natively understands clothing types — no body-split hacks needed.
DEEPFASHION2_LABELS = {
    0:  "short sleeved shirt",
    1:  "long sleeved shirt",
    2:  "short sleeved outwear",
    3:  "long sleeved outwear",
    4:  "vest",
    5:  "sling",
    6:  "shorts",
    7:  "trousers",
    8:  "skirt",
    9:  "short sleeved dress",
    10: "long sleeved dress",
    11: "vest dress",
    12: "sling dress",
}

# Minimum confidence for a detection to be shown to the user
DETECTION_CONFIDENCE = 0.30

# Minimum pixel area for a detection box (filters out tiny false positives)
MIN_BBOX_AREA = 1500  # pixels squared


class LocusVisualizer:
    def __init__(self):
        print("Loading DeepFashion2 detection model from HuggingFace...")
        # This downloads ~22MB on first run and caches it automatically.
        # It's a YOLOv8s model fine-tuned on 491K fashion images — it directly
        # detects clothing categories without any body-split hacks.
        model_path = hf_hub_download(
            repo_id="Bingsu/adetailer",
            filename="deepfashion2_yolov8s-seg.pt"
        )
        self.yolo = YOLO(model_path)
        print("DeepFashion2 model loaded!")

        print("Loading CLIP embedding model...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        print("Pre-loading background remover (rembg)...")
        self.rembg_session = new_session("u2net")

        # CLIP labels for fine-grained vectorization and category filtering.
        # These map from DeepFashion2's detection labels to your search categories.
        self.clip_labels = [
            "dress", "pants", "jeans", "shirt", "t-shirt",
            "jacket", "coat", "shoes", "sneakers", "bag",
            "handbag", "skirt", "shorts", "hat", "glasses", "watch"
        ]

        # Pre-compute CLIP text embeddings once at startup
        print("Pre-computing CLIP text embeddings...")
        text_inputs = self.clip_processor(
            text=self.clip_labels, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            self.text_features = self.clip_model.get_text_features(**text_inputs)
            self.text_features /= self.text_features.norm(p=2, dim=-1, keepdim=True)

        print("Locus Visual Engine Ready! (DeepFashion2 + CLIP)")

    # -------------------------------------------------------------------------
    # PUBLIC METHOD 1: detect_objects()
    # -------------------------------------------------------------------------
    def detect_objects(self, image_bytes):
        """
        Uses the DeepFashion2 YOLOv8 model to detect clothing items directly.

        This replaces the old approach of:
          - Generic YOLO detecting "person" → splitting into body regions → CLIP guessing

        New approach:
          - DeepFashion2 YOLO directly detects "long_sleeved_shirt", "trousers",
            "skirt" etc. with precise bounding boxes and segmentation masks.
          - CLIP is then used to get the exact label for the detected region,
            consistent with your search index labels.

        Returns: list of dicts with bbox [x1,y1,x2,y2], label, and score.
        """
        t0 = time.time()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            W, H = image.size

            # Run DeepFashion2 detection
            results = self.yolo(image, conf=DETECTION_CONFIDENCE, verbose=False)[0]

            detections = []

            for box in results.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Filter out detections that are too small
                area = (x2 - x1) * (y2 - y1)
                if area < MIN_BBOX_AREA:
                    continue

                # Get the DeepFashion2 label (e.g. "long sleeved shirt")
                df2_label = DEEPFASHION2_LABELS.get(class_id, "clothing")

                # Crop the detected region and run CLIP for a consistent label
                # that matches your search index categories
                crop = image.crop((x1, y1, x2, y2))
                clip_label, clip_conf = self._classify_crop(crop)

                # Use DeepFashion2's label as display name (more descriptive),
                # CLIP label as the search category filter
                detections.append({
                    "bbox":         [x1, y1, x2, y2],
                    "label":        df2_label,       # shown to user: "long sleeved shirt"
                    "search_label": clip_label,      # used for search filter: "shirt"
                    "score":        round(conf, 3)
                })

            # ── Fallback: flat-lay or product photo with no person ────────────
            # DeepFashion2 still works on flat-lay product photos, but if it
            # finds nothing (e.g. a bag, shoe, or accessory not in its 13
            # clothing categories), fall back to full-image CLIP classification.
            if not detections:
                print("DeepFashion2 found nothing. Running full-image CLIP fallback.")
                clip_label, clip_conf = self._classify_crop(image)
                if clip_conf >= 0.35:
                    detections.append({
                        "bbox":         [0, 0, W, H],
                        "label":        clip_label,
                        "search_label": clip_label,
                        "score":        round(clip_conf, 3)
                    })

            print(f"Detected {len(detections)} items in {(time.time()-t0):.2f}s")
            return detections, W, H

        except Exception as e:
            print(f"Detection error: {e}")
            return [], 0, 0

    # -------------------------------------------------------------------------
    # PUBLIC METHOD 2: process_image()
    # -------------------------------------------------------------------------
    def process_image(self, image_bytes):
        """
        Full pipeline for a single selected item:
        1. Remove background (rembg)
        2. Ghost image check
        3. Smart crop to content bounding box
        4. CLIP vectorization (produces the 512-dim search vector)
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

            # Resize before rembg — biggest speed bottleneck
            if max(input_image.size) > 512:
                input_image.thumbnail((512, 512))
                print(f"Resized {original_size} -> {input_image.size}")

            print("Removing background...")
            output_image = remove(input_image, session=self.rembg_session)

            # Ghost image check: rembg removed everything → reject
            alpha_max = output_image.getextrema()[3][1]
            if alpha_max == 0:
                print("Ghost image detected. Rejecting.")
                return None, None, None

            # Crop tightly to just the item content
            bbox = output_image.getbbox()
            if bbox:
                output_image = output_image.crop(bbox)

            # Paste onto white background (CLIP expects RGB not RGBA)
            white_bg = Image.new("RGB", output_image.size, (255, 255, 255))
            white_bg.paste(output_image, mask=output_image.split()[3])

            # CLIP vectorization → produces the 512-dim embedding for Qdrant
            clip_inputs = self.clip_processor(images=white_bg, return_tensors="pt")
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**clip_inputs)

            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
            vector = image_features[0].tolist()

            # Category classification with 45% confidence threshold
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

            # Encode debug image as base64 for the UI
            buf = io.BytesIO()
            white_bg.save(buf, format="PNG")
            debug_img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            print(f"process_image() done in {(time.time()-t0):.2f}s")
            return vector, detected_category, debug_img_b64

        except Exception as e:
            print(f"process_image() error: {e}")
            return None, None, None

    # -------------------------------------------------------------------------
    # PRIVATE: _classify_crop()
    # -------------------------------------------------------------------------
    def _classify_crop(self, pil_image):
        """
        Runs CLIP zero-shot classification on a PIL image.
        Returns (best_label, confidence_score) from self.clip_labels.
        """
        clip_inputs = self.clip_processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**clip_inputs)
        image_features /= image_features.norm(p=2, dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        top_score, top_idx = similarity[0].topk(1)

        return self.clip_labels[top_idx[0]], top_score[0].item()