# =============================================================================
# vectorizer.py
#
# Clean separation of concerns:
#   YOLO  → detection only   (finds items, draws bounding boxes)
#   CLIP  → classification + embedding  (what is it? what does it look like?)
#
# process_image() returns 4 values:
#   vector, category_tag, category_scores, debug_img_b64
#
# CHANGES vs previous version:
#   - Added _iou() and _nms() helpers at module level
#   - detect_objects() now runs NMS after merging both detectors
#     to eliminate duplicate boxes from overlapping model coverage
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


# =============================================================================
# NMS HELPERS  (module-level, not inside the class)
# =============================================================================

def _iou(a, b):
    """
    Intersection over Union for two bounding boxes [x1, y1, x2, y2].
    Returns a float between 0.0 (no overlap) and 1.0 (identical boxes).
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def _nms(detections, iou_threshold=0.45):
    """
    Greedy Non-Maximum Suppression across all detections regardless of source model.

    Algorithm:
        1. Sort all boxes by confidence score (highest first)
        2. Keep the top box
        3. Suppress any other box that overlaps it by more than iou_threshold
        4. Repeat for remaining boxes

    This eliminates duplicates when both DeepFashion2 AND YOLOS fire on the same item.
    """
    if not detections:
        return []

    dets = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []
    suppressed = set()

    for i, d in enumerate(dets):
        if i in suppressed:
            continue
        kept.append(d)
        for j in range(i + 1, len(dets)):
            if j in suppressed:
                continue
            if _iou(d["bbox"], dets[j]["bbox"]) > iou_threshold:
                suppressed.add(j)

    print(f"  NMS: {len(detections)} -> {len(kept)} detections after suppression")
    return kept


# =============================================================================
# MAIN CLASS
# =============================================================================

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
        Runs both detection models on the same image, merges results, and
        applies cross-model NMS to remove duplicate boxes.

        Flow:
            1. ClothingDetector  -> shirts, pants, dresses, skirts, jackets
            2. AccessoryDetector -> sweaters, coats, jumpsuits, shoes, bags, etc.
            3. Merge all results
            4. NMS — suppress overlapping boxes, keep highest confidence
            5. Fallback to full-image CLIP if nothing survives

        Returns:
            (detections, image_width, image_height)
        """
        t0 = time.time()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            W, H  = image.size

            clothing    = self.clothing_detector.detect(image, self._classify_crop)
            accessories = self.accessory_detector.detect(image, self._classify_crop)

            # Merge then deduplicate with NMS
            all_detections = _nms(clothing + accessories, iou_threshold=0.45)

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
    def process_image(self, image_bytes, skip_rembg=False, yolo_label=""):
        """
        Full pipeline for a single item.

        Steps:
            1. Resize if needed (max 512px)
            2a. skip_rembg=True  -> bounding box crop, skip background removal
            2b. skip_rembg=False -> rembg background removal
            3. CLIP vectorization -> 512-dim embedding
            4. CLIP classification -> all 15 scores (softmax)
               top score >= 0.45 -> category_tag = that label
               top score <  0.45 -> category_tag = None (indexed without filter)

        Args:
            image_bytes: raw bytes of the image
            skip_rembg:  True when called with a tight bounding box crop
            yolo_label:  canonical label passed from gateway (overrides CLIP if set)

        Returns:
            (vector, category_tag, category_scores, debug_img_b64)
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
                print(f"Resized {original_size} -> {input_image.size}")

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

            category_scores = {
                label: round(scores_tensor[i].item(), 4)
                for i, label in enumerate(self.clip_labels)
            }

            # If YOLO passed a label, trust it over CLIP (YOLO is the specialist)
            if yolo_label and yolo_label.strip() in self.clip_labels:
                category_tag = yolo_label.strip()
                print(f"Category from YOLO label: '{category_tag}'")
            else:
                clip_conf  = scores_tensor.max().item()
                clip_label = self.clip_labels[scores_tensor.argmax().item()]
                if clip_conf >= 0.45:
                    category_tag = clip_label
                    print(f"CLIP classification: '{category_tag}' ({clip_conf:.2f})")
                else:
                    category_tag = None
                    print(f"CLIP low confidence: '{clip_label}' ({clip_conf:.2f}) -> no category filter")

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