import torch
import io
import base64
import time
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from rembg import remove, new_session
from ultralytics import YOLO

# ─── YOLO class IDs relevant to fashion ──────────────────────────────────────
FASHION_RELEVANT_YOLO_CLASSES = {
    24: "handbag",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    0:  "person",
}

# Body region splits — NO OVERLAP between top and bottom.
# Each region is tight and non-overlapping to avoid detecting the same
# item twice. The key insight: we use rembg on each crop BEFORE asking
# CLIP what it is, so CLIP sees just the clothing item, not the whole body.
BODY_REGION_SPLITS = {
    "top":    (0.05, 0.52),  # torso only — shirt, jacket, dress
    "bottom": (0.50, 0.88),  # waist to ankle — pants, skirt
    "feet":   (0.85, 1.00),  # ankles down — shoes
}

# Minimum pixel dimensions for a crop to be worth classifying
MIN_CROP_W = 40
MIN_CROP_H = 40

# CLIP confidence threshold for body region crops
BODY_REGION_MIN_CONFIDENCE = 0.32


class LocusVisualizer:
    def __init__(self):
        print("Loading YOLO detection model...")
        self.yolo = YOLO("yolov8n.pt")

        print("Loading CLIP embedding model...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        print("Pre-loading background remover (rembg)...")
        self.rembg_session = new_session("u2net")

        # Fine-grained fashion labels CLIP classifies into.
        # YOLO finds WHERE. CLIP identifies WHAT.
        self.labels = [
            "dress", "pants", "jeans", "shirt", "t-shirt",
            "jacket", "coat", "shoes", "sneakers", "bag",
            "handbag", "skirt", "shorts", "hat", "glasses", "watch"
        ]

        # Pre-compute text embeddings once at startup to save ~100ms per call
        print("Pre-computing CLIP text embeddings...")
        text_inputs = self.clip_processor(text=self.labels, return_tensors="pt", padding=True)
        with torch.no_grad():
            self.text_features = self.clip_model.get_text_features(**text_inputs)
            self.text_features /= self.text_features.norm(p=2, dim=-1, keepdim=True)

        print("Locus Visual Engine Ready! (YOLO + CLIP)")

    # -------------------------------------------------------------------------
    # PUBLIC METHOD 1: detect_objects()
    # -------------------------------------------------------------------------
    def detect_objects(self, image_bytes):
        """
        Two-stage detection pipeline:

        Stage 1 — YOLO: finds WHERE objects are (bounding boxes).
            - Direct fashion items (bags, ties): returned as-is.
            - People: split into 3 non-overlapping body regions.

        Stage 2 — CLIP: identifies WHAT each region is.
            - Each crop is background-removed first so CLIP sees just the
              clothing item, not surrounding body/background noise.
            - Low-confidence results are filtered out.

        Deduplication: after all regions are collected, we remove any
        two detections whose boxes overlap more than 40% (IoU-based).
        """
        t0 = time.time()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            W, H = image.size

            results = self.yolo(image, conf=0.25, verbose=False)[0]
            raw_detections = []

            for box in results.boxes:
                class_id = int(box.cls[0])
                yolo_conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # ── Direct fashion accessory ──────────────────────────────────
                if class_id in FASHION_RELEVANT_YOLO_CLASSES and class_id != 0:
                    crop = image.crop((x1, y1, x2, y2))
                    clip_label, clip_conf = self._classify_crop(crop)
                    raw_detections.append({
                        "bbox":  [x1, y1, x2, y2],
                        "label": clip_label,
                        "score": round(yolo_conf, 3)
                    })

                # ── Person: split into non-overlapping body regions ───────────
                elif class_id == 0:
                    person_h = y2 - y1
                    person_w = x2 - x1

                    for region_name, (frac_top, frac_bot) in BODY_REGION_SPLITS.items():
                        ry1 = y1 + int(person_h * frac_top)
                        ry2 = y1 + int(person_h * frac_bot)

                        # Skip regions too small to be meaningful
                        if (ry2 - ry1) < MIN_CROP_H or person_w < MIN_CROP_W:
                            continue

                        crop = image.crop((x1, ry1, x2, ry2))

                        # KEY FIX: remove background before classifying.
                        # Without this, CLIP sees "person wearing a shirt"
                        # instead of "a shirt", and gets confused.
                        clean_crop = self._remove_bg_for_classification(crop)
                        clip_label, clip_conf = self._classify_crop(clean_crop)

                        if clip_conf >= BODY_REGION_MIN_CONFIDENCE:
                            raw_detections.append({
                                "bbox":  [x1, ry1, x2, ry2],
                                "label": clip_label,
                                "score": round(clip_conf, 3)
                            })

            # ── Fallback: no person or accessory found ────────────────────────
            # Handles flat-lay product photos where YOLO finds nothing relevant.
            if not raw_detections:
                print("YOLO found nothing relevant. Running full-image fallback.")
                clean_image = self._remove_bg_for_classification(image)
                clip_label, clip_conf = self._classify_crop(clean_image)
                if clip_conf >= 0.35:
                    raw_detections.append({
                        "bbox":  [0, 0, W, H],
                        "label": clip_label,
                        "score": round(clip_conf, 3)
                    })

            # ── Deduplicate overlapping boxes ─────────────────────────────────
            final_detections = self._nms(raw_detections, iou_threshold=0.40)

            print(f"Detected {len(final_detections)} items in {(time.time()-t0):.2f}s")
            return final_detections, W, H

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
        3. Smart crop to content
        4. CLIP vectorization
        5. Category classification with confidence threshold
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
            best_label = self.labels[top_idx[0]]

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

    # -------------------------------------------------------------------------
    # PRIVATE: _remove_bg_for_classification()
    # -------------------------------------------------------------------------
    def _remove_bg_for_classification(self, pil_image):
        """
        Lightweight background removal specifically for CLIP classification.

        Why: when CLIP receives a crop of "upper body of a person wearing a
        shirt", it sees the face, arms, background — all noise. Removing the
        background first means CLIP sees approximately just the clothing item,
        which dramatically improves label accuracy.

        We resize to 256px first (much smaller than the 512px used in
        process_image) because we only need good enough quality for
        classification, not for vectorization. This keeps detection fast.
        """
        try:
            # Resize small for speed — classification doesn't need full resolution
            img = pil_image.copy()
            if max(img.size) > 256:
                img.thumbnail((256, 256))

            img_rgba = img.convert("RGBA")
            removed = remove(img_rgba, session=self.rembg_session)

            # Check if rembg removed everything (ghost image)
            alpha_max = removed.getextrema()[3][1]
            if alpha_max == 0:
                return pil_image  # Fall back to original if rembg failed

            # Paste onto white background for CLIP
            white_bg = Image.new("RGB", removed.size, (255, 255, 255))
            white_bg.paste(removed, mask=removed.split()[3])
            return white_bg

        except Exception:
            return pil_image  # Always fall back gracefully

    # -------------------------------------------------------------------------
    # PRIVATE: _classify_crop()
    # -------------------------------------------------------------------------
    def _classify_crop(self, pil_image):
        """
        Runs CLIP zero-shot classification on a PIL image.
        Returns (best_label, confidence_score).
        """
        clip_inputs = self.clip_processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**clip_inputs)
        image_features /= image_features.norm(p=2, dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        top_score, top_idx = similarity[0].topk(1)

        return self.labels[top_idx[0]], top_score[0].item()

    # -------------------------------------------------------------------------
    # PRIVATE: _nms()
    # -------------------------------------------------------------------------
    def _nms(self, detections, iou_threshold=0.40):
        """
        Non-Maximum Suppression: removes duplicate overlapping boxes.

        Sorts by confidence score (highest first), keeps the best box,
        then removes any other box that overlaps it by more than iou_threshold.
        Repeats until no boxes remain.
        """
        if not detections:
            return []

        detections = sorted(detections, key=lambda x: x["score"], reverse=True)
        kept = []

        while detections:
            best = detections.pop(0)
            kept.append(best)
            detections = [
                d for d in detections
                if self._iou(best["bbox"], d["bbox"]) < iou_threshold
            ]

        return kept

    # -------------------------------------------------------------------------
    # PRIVATE: _iou()
    # -------------------------------------------------------------------------
    def _iou(self, boxA, boxB):
        """
        Intersection over Union: 0 = no overlap, 1 = identical boxes.
        Used by NMS to decide if two detections are the same object.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(areaA + areaB - interArea)