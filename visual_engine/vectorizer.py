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

            clothing    = self.clothing_detector.detect(image, self._classify_crop)
            accessories = self.accessory_detector.detect(image, self._classify_crop)
            all_detections = clothing + accessories

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
    def process_image(self, image_bytes, skip_rembg=False, yolo_label=""):
        """
        Full pipeline for a single selected item:

        1. Resize if needed
        2a. If skip_rembg=True  → image is already a tight bounding box crop,
                                   go straight to CLIP (no background removal)
        2b. If skip_rembg=False → run rembg background removal (full-image fallback)
        3. CLIP vectorization (512-dim vector for Qdrant)
        4. Category:
              - If yolo_label provided → use it directly (YOLO already identified the item)
              - Otherwise → CLIP zero-shot classification with 45% confidence threshold

        Args:
            image_bytes: raw image bytes
            skip_rembg:  True when image comes from a bounding box crop
            yolo_label:  YOLO's detected label — bypasses CLIP classification when set
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

            if skip_rembg:
                # Bounding box crop — background already minimised, skip rembg
                print("Skipping background removal (bounding box crop)")
                white_bg = Image.new("RGB", input_image.size, (255, 255, 255))
                if input_image.mode == "RGBA":
                    white_bg.paste(input_image, mask=input_image.split()[3])
                else:
                    white_bg.paste(input_image)
            else:
                # Full image — run rembg to isolate the clothing item
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

            # ── CLIP Vectorization ────────────────────────────────────────────
            clip_inputs = self.clip_processor(images=white_bg, return_tensors="pt")
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**clip_inputs)

            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
            vector = image_features[0].tolist()

            # ── Category Classification ───────────────────────────────────────
            category_confidence = 1.0  # Default to 100% if YOLO found it

            if yolo_label and yolo_label.strip():
                # YOLO already identified this item with a bounding box —
                # trust it over CLIP which sees the whole crop including background
                detected_category = yolo_label.strip()
                print(f"Category from YOLO: {detected_category}")
            else:
                # No YOLO label — fall back to CLIP zero-shot classification
                similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
                top_score, top_idx = similarity[0].topk(1)
                
                category_confidence = top_score[0].item() # 👈 NEW: Capture the score!
                best_label = self.clip_labels[top_idx[0]]
                if category_confidence < 0.45:
                    print(f"Low confidence ({category_confidence:.2f}) for '{best_label}'. No category filter.")
                    detected_category = None
                else:
                    detected_category = best_label
                    print(f"Category from CLIP: {detected_category} ({category_confidence:.2f})")

            # ── Debug Image ───────────────────────────────────────────────────
            buf = io.BytesIO()
            white_bg.save(buf, format="PNG")
            debug_img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            print(f"process_image() done in {(time.time()-t0):.2f}s")
            return vector, detected_category, category_confidence, debug_img_b64

        except Exception as e:
            print(f"process_image() error: {e}")
            # 👈 NEW: Return 4 Nones instead of 3
            return None, None, None, None

    # =========================================================================
    # PRIVATE: _classify_crop()
    # Shared CLIP utility — passed into both detectors as classify_fn
    # =========================================================================
    def _classify_crop(self, pil_image):
        """
        Runs CLIP zero-shot classification on a PIL image.
        Returns (best_label, confidence) from self.clip_labels.
        Used by both detectors to get a consistent search label.
        """
        clip_inputs = self.clip_processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**clip_inputs)
        image_features /= image_features.norm(p=2, dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        top_score, top_idx = similarity[0].topk(1)

        return self.clip_labels[top_idx[0]], top_score[0].item()