# =============================================================================
# vectorizer.py
#
# Clean separation of concerns:
#   YOLO  → detection only   (finds items, draws bounding boxes)
#   CLIP  → classification + embedding  (what is it? what does it look like?)
#
# process_image() returns 4 values:
#   vector, category_tag, category_scores, debug_img_b64
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
from clip_labels import CANONICAL_LABELS


class LocusVisualizer:
    def __init__(self):

        # ── Detection models ──────────────────────────────────────────────────
        self.clothing_detector   = ClothingDetector()
        self.accessory_detector  = AccessoryDetector()

        # ── CLIP ──────────────────────────────────────────────────────────────
        print("Loading CLIP (Vectorization & Classification)")
        self.clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        # ── rembg ─────────────────────────────────────────────────────────────
        print("Loading rembg (Background Removal)")
        self.rembg_session = new_session("u2net")

        # ── Labels ────────────────────────────────────────────────────────────
        self.clip_labels = CANONICAL_LABELS   # 15 canonical labels from clip_labels.py

        # Pre-compute text embeddings once at startup — shape (15, 512)
        # Every classification call is then just one matrix multiply (fast)
        print(f"Pre-computing CLIP text embeddings for {len(self.clip_labels)} labels...")
        text_inputs = self.clip_processor(
            text=self.clip_labels, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            self.text_features = self.clip_model.get_text_features(**text_inputs)
            self.text_features /= self.text_features.norm(p=2, dim=-1, keepdim=True)

        print("=" * 50)
        print("LOCUS VISUAL ENGINE READY")
        print(f"Labels ({len(self.clip_labels)}): {self.clip_labels}")
        print("=" * 50)

    # =========================================================================
    # PUBLIC METHOD 1: detect_objects()
    # Called by gateway /detect endpoint.
    # =========================================================================
    def detect_objects(self, image_bytes):
        """
        Runs both detection models on the same image and merges results.

        Flow:
            1. ClothingDetector  → shirts, pants, dresses, skirts, jackets
            2. AccessoryDetector → sweaters, coats, jumpsuits, shoes, bags, etc.
            3. Results merged
            4. Fallback to full-image CLIP if both models find nothing

        Returns:
            (detections, image_width, image_height)
        """
        t0 = time.time()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            W, H  = image.size

            clothing       = self.clothing_detector.detect(image, self._classify_crop)
            accessories    = self.accessory_detector.detect(image, self._classify_crop)
            all_detections = clothing + accessories

            if not all_detections:
                print("Both models found nothing — running full-image CLIP fallback.")
                label, conf, _ = self._classify_crop(image)
                if conf >= 0.35:
                    all_detections.append({
                        "bbox":         [0, 0, W, H],
                        "label":        label,
                        "search_label": label,
                        "score":        round(conf, 3),
                        "source":       "clip_fallback"
                    })

            print(f"Total: {len(all_detections)} detections in {(time.time()-t0):.2f}s")
            return all_detections, W, H

        except Exception as e:
            print(f"detect_objects() error: {e}")
            return [], 0, 0

    # =========================================================================
    # PUBLIC METHOD 2: process_image()
    # Called by gateway /search and /add-bulk endpoints.
    # =========================================================================
    def process_image(self, image_bytes, skip_rembg=False):
        """
        Full pipeline for a single item.

        Steps:
            1. Resize if needed (max 512px)
            2a. skip_rembg=True  → bounding box crop, skip background removal
            2b. skip_rembg=False → rembg background removal
            3. CLIP vectorization → 512-dim embedding
            4. CLIP classification → all 15 scores (softmax)
               top score ≥ 0.45 → category_tag = that label
               top score < 0.45 → category_tag = None (indexed without filter)

        Args:
            image_bytes: raw bytes of the image
            skip_rembg:  True when called with a tight bounding box crop

        Returns:
            (vector, category_tag, category_scores, debug_img_b64)
            vector:          list[float] — 512-dim CLIP embedding
            category_tag:    str | None  — top label if conf ≥ 0.45, else None
            category_scores: dict        — all 15 scores {"shirt": 0.02, "dress": 0.91, ...}
            debug_img_b64:   str         — base64 PNG of preprocessed image
        """
        t0 = time.time()
        try:
            try:
                input_image   = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
                original_size = input_image.size
            except Exception:
                print("Not a valid image file.")
                return None, None, None, None

            # ── Step 1: Resize ────────────────────────────────────────────────
            if max(input_image.size) > 512:
                input_image.thumbnail((512, 512))
                print(f"Resized {original_size} → {input_image.size}")

            # ── Step 2: Background handling ───────────────────────────────────
            if skip_rembg:
                print("Skipping background removal (bounding box crop)")
                white_bg = Image.new("RGB", input_image.size, (255, 255, 255))
                if input_image.mode == "RGBA":
                    white_bg.paste(input_image, mask=input_image.split()[3])
                else:
                    white_bg.paste(input_image)
            else:
                print("Removing background...")
                output_image = remove(input_image, session=self.rembg_session)

                alpha_max = output_image.getextrema()[3][1]
                if alpha_max == 0:
                    print("Ghost image detected — rejecting.")
                    return None, None, None, None

                bbox = output_image.getbbox()
                if bbox:
                    output_image = output_image.crop(bbox)

                white_bg = Image.new("RGB", output_image.size, (255, 255, 255))
                white_bg.paste(output_image, mask=output_image.split()[3])

            # ── Step 3: CLIP Vectorization ────────────────────────────────────
            clip_inputs = self.clip_processor(images=white_bg, return_tensors="pt")
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**clip_inputs)
            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
            vector = image_features[0].tolist()

            # ── Step 4: CLIP Classification ───────────────────────────────────
            similarity    = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            scores_tensor = similarity[0]   # shape: (15,)

            # All 15 scores stored in Qdrant — useful for future re-classification
            # without re-running the visual engine
            category_scores = {
                label: round(scores_tensor[i].item(), 4)
                for i, label in enumerate(self.clip_labels)
            }

            clip_conf  = scores_tensor.max().item()
            clip_label = self.clip_labels[scores_tensor.argmax().item()]

            if clip_conf >= 0.45:
                category_tag = clip_label
                print(f"CLIP classification: '{category_tag}' ({clip_conf:.2f})")
            else:
                category_tag = None
                print(f"CLIP low confidence: '{clip_label}' ({clip_conf:.2f}) → no category filter")

            # ── Step 5: Debug image ───────────────────────────────────────────
            buf = io.BytesIO()
            white_bg.save(buf, format="PNG")
            debug_img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            print(f"process_image() done in {(time.time()-t0):.2f}s")
            return vector, category_tag, category_scores, debug_img_b64

        except Exception as e:
            print(f"process_image() error: {e}")
            return None, None, None, None

    # =========================================================================
    # PRIVATE: _classify_crop()
    # Passed into both detectors as classify_fn (used by fallback path).
    # =========================================================================
    def _classify_crop(self, pil_image):
        """
        CLIP zero-shot classification on a PIL image.

        Returns:
            (best_label, best_confidence, all_scores)
        """
        clip_inputs = self.clip_processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**clip_inputs)
        image_features /= image_features.norm(p=2, dim=-1, keepdim=True)

        similarity    = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        scores_tensor = similarity[0]

        best_idx   = scores_tensor.argmax().item()
        best_conf  = scores_tensor[best_idx].item()
        best_label = self.clip_labels[best_idx]

        all_scores = {
            label: round(scores_tensor[i].item(), 4)
            for i, label in enumerate(self.clip_labels)
        }

        return best_label, best_conf, all_scores