# =============================================================================
# vectorizer.py
#
# CHANGES vs previous version:
#   - clip_labels imported from clip_labels.py (single source of truth)
#   - _classify_crop() now returns ALL 15 scores, not just top-1
#   - process_image() accepts yolo_conf parameter (YOLO bbox confidence)
#   - Confidence ensemble implemented:
#       YOLO canonical label + bbox conf  vs  CLIP top label + softmax conf
#       Whichever is higher confidence wins → becomes category_tag
#   - category_scores (all 15 label scores) now returned and stored in Qdrant
#   - process_image() always returns 4 values: vector, category_tag,
#     category_scores, debug_img_b64
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
from clip_labels import CANONICAL_LABELS   # ← single source of truth


class LocusVisualizer:
    def __init__(self):

        # ── Detection Model 1 ─────────────────────────────────────────────────
        self.clothing_detector = ClothingDetector()

        # ── Detection Model 2 ─────────────────────────────────────────────────
        self.accessory_detector = AccessoryDetector()

        # ── CLIP ──────────────────────────────────────────────────────────────
        print("Loading CLIP (Vectorization & Classification)")
        self.clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        # ── rembg ─────────────────────────────────────────────────────────────
        print("Loading rembg (Background Removal)")
        self.rembg_session = new_session("u2net")

        # ── Labels — imported, never hardcoded here ───────────────────────────
        # CANONICAL_LABELS is the 15-label list from clip_labels.py.
        # If you add a label there, it automatically applies here at next restart.
        self.clip_labels = CANONICAL_LABELS

        # Pre-compute text embeddings once at startup.
        # Shape: (15, 512) — one 512-dim vector per label.
        # Stays in memory so every classify call is just a matrix multiply.
        print(f"Pre-computing CLIP text embeddings for {len(self.clip_labels)} labels...")
        text_inputs = self.clip_processor(
            text=self.clip_labels, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            self.text_features = self.clip_model.get_text_features(**text_inputs)
            self.text_features /= self.text_features.norm(p=2, dim=-1, keepdim=True)

        print("=" * 50)
        print("LOCUS VISUAL ENGINE READY")
        print(f"Labels: {self.clip_labels}")
        print("=" * 50)

    # =========================================================================
    # PUBLIC METHOD 1: detect_objects()
    # Called by gateway /detect endpoint.
    # Runs both detectors, merges results, falls back to CLIP if nothing found.
    # =========================================================================
    def detect_objects(self, image_bytes):
        """
        Runs both detection models independently on the same image.

        Flow:
            1. ClothingDetector  → shirts, pants, dresses, skirts, jackets
            2. AccessoryDetector → sweaters, coats, jumpsuits, shoes, bags, etc.
            3. Results merged (simple concatenation, no cross-model deduplication)
            4. Fallback to full-image CLIP if both models find nothing

        Returns:
            (detections, image_width, image_height)
            detections: list of dicts with bbox, label, search_label, score, source
        """
        t0 = time.time()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            W, H  = image.size

            clothing    = self.clothing_detector.detect(image, self._classify_crop)
            accessories = self.accessory_detector.detect(image, self._classify_crop)
            all_detections = clothing + accessories

            if not all_detections:
                print("Both models found nothing. Running full-image CLIP fallback.")
                label, conf, _ = self._classify_crop(image)   # _ = all_scores, unused here
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
    # Embeds the item and classifies it using the confidence ensemble.
    # =========================================================================
    def process_image(self, image_bytes, skip_rembg=False, yolo_label="", yolo_conf=0.0):
        """
        Full pipeline for a single selected item.

        Steps:
            1. Resize if needed (max 512px)
            2a. skip_rembg=True  → bounding box crop, skip background removal
            2b. skip_rembg=False → run rembg to isolate item from background
            3. CLIP vectorization → 512-dim embedding for Qdrant
            4. CLIP classification → all 15 label scores (softmax)
            5. Confidence ensemble:
                  Branch A: yolo_label + yolo_conf  (YOLO bbox detection confidence)
                  Branch B: clip_label + clip_conf  (CLIP softmax top score)
                  Winner:   whichever branch has higher confidence → category_tag
            6. Return category_scores (all 15 scores) for Qdrant payload storage

        Args:
            image_bytes: raw bytes of the image to process
            skip_rembg:  True when image is a tight bounding box crop
            yolo_label:  canonical label from YOLO (already mapped via YOLO_TO_CANONICAL
                         or FASHIONPEDIA_TO_CANONICAL). Empty string if no YOLO detection.
            yolo_conf:   YOLO bounding box confidence score (0.0-1.0).
                         Only meaningful when yolo_label is set.

        Returns:
            (vector, category_tag, category_scores, debug_img_b64)
            vector:          list[float] — 512-dim CLIP embedding
            category_tag:    str | None  — winning canonical label, None if low confidence
            category_scores: dict        — all 15 scores e.g. {"dress": 0.91, "shirt": 0.03, ...}
            debug_img_b64:   str         — base64 PNG of preprocessed image (for UI debugger)
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
                    print("Ghost image detected. Rejecting.")
                    return None, None, None, None

                bbox = output_image.getbbox()
                if bbox:
                    output_image = output_image.crop(bbox)

                white_bg = Image.new("RGB", output_image.size, (255, 255, 255))
                white_bg.paste(output_image, mask=output_image.split()[3])

            # ── Step 3: CLIP Vectorization ────────────────────────────────────
            # Produces the 512-dim embedding stored in Qdrant for similarity search.
            clip_inputs = self.clip_processor(images=white_bg, return_tensors="pt")
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**clip_inputs)
            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
            vector = image_features[0].tolist()

            # ── Step 4: CLIP Classification (all 15 scores) ───────────────────
            # Always run this even when YOLO gave us a label.
            # Two reasons:
            #   (a) We need CLIP's confidence to compare against YOLO's in the ensemble
            #   (b) We store ALL 15 scores in Qdrant — useful for future re-classification
            similarity    = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            scores_tensor = similarity[0]                          # shape: (15,)

            category_scores = {
                label: round(scores_tensor[i].item(), 4)
                for i, label in enumerate(self.clip_labels)
            }

            clip_conf_val  = scores_tensor.max().item()
            clip_label_val = self.clip_labels[scores_tensor.argmax().item()]

            # ── Step 5: Confidence Ensemble ───────────────────────────────────
            #
            # Branch A (YOLO): canonical label from bbox detection + bbox confidence
            # Branch B (CLIP): top softmax label + softmax confidence
            #
            # Both are 0-1 floats → directly comparable.
            # If neither clears MIN_CONF → category_tag = None
            # (product still indexed, just without a category filter)

            MIN_CONF       = 0.45
            yolo_available = bool(yolo_label and yolo_label.strip())

            if yolo_available and yolo_conf >= clip_conf_val:
                # YOLO is more confident — trust the detector
                category_tag = yolo_label.strip()
                winner       = "YOLO"
                win_conf     = yolo_conf
            elif clip_conf_val >= MIN_CONF:
                # CLIP is more confident (or YOLO wasn't available)
                category_tag = clip_label_val
                winner       = "CLIP"
                win_conf     = clip_conf_val
            else:
                # Neither branch is confident enough
                category_tag = None
                winner       = "none"
                win_conf     = max(yolo_conf, clip_conf_val)

            print(
                f"Ensemble → "
                f"YOLO: '{yolo_label}' ({yolo_conf:.2f})  |  "
                f"CLIP: '{clip_label_val}' ({clip_conf_val:.2f})  |  "
                f"Winner: {winner} → '{category_tag}' ({win_conf:.2f})"
            )

            # ── Step 6: Debug image ───────────────────────────────────────────
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
    # Passed into detectors as classify_fn.
    # Returns all 15 scores now (not just top-1) so callers have full info.
    # =========================================================================
    def _classify_crop(self, pil_image):
        """
        Runs CLIP zero-shot classification on a PIL image crop.

        Returns:
            (best_label, best_confidence, all_scores)
            best_label:      str   — canonical label with highest softmax score
            best_confidence: float — that label's softmax score
            all_scores:      dict  — full {label: score} dict for all 15 labels
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