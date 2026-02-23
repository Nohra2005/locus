import torch
import io
import base64
import time
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from rembg import remove, new_session
from ultralytics import YOLO

# ─── YOLO class IDs that are relevant to fashion ─────────────────────────────
# YOLO is trained on 80 COCO classes. We only care about the ones that overlap
# with fashion items. Everything else (car, dog, chair...) gets ignored.
#
# Full COCO class list: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
FASHION_RELEVANT_YOLO_CLASSES = {
    24: "handbag",
    25: "umbrella",   # catches parasols / beach bags sometimes
    26: "handbag",
    27: "tie",
    28: "suitcase",
    # Person is class 0 — we detect the person, then use their bounding box
    # to know WHERE to look, but we don't return "person" as an item.
    0:  "person",
}

# If YOLO finds a "person", we use these sub-regions of their bounding box
# to isolate the clothing they're wearing (torso, legs, feet).
# These are fractions of the person's bounding box height.
BODY_REGION_SPLITS = {
    "top":    (0.0,  0.5),   # upper half -> shirt, jacket, dress top
    "bottom": (0.45, 0.85),  # lower half -> pants, skirt, jeans
    "feet":   (0.80, 1.0),   # bottom 20% -> shoes
}


class LocusVisualizer:
    def __init__(self):
        print("Loading YOLO detection model...")
        # yolov8n = "nano" — the smallest, fastest YOLOv8 variant.
        # It downloads automatically (~6MB) on first run.
        # "n" = nano, "s" = small, "m" = medium. Nano is fine for CPU.
        self.yolo = YOLO("yolov8n.pt")

        print("Loading CLIP embedding model...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        print("Pre-loading background remover (rembg)...")
        self.rembg_session = new_session("u2net")

        # These are the fine-grained fashion categories CLIP classifies into.
        # YOLO finds WHERE items are. CLIP identifies WHAT they are.
        self.labels = [
            "dress", "pants", "jeans", "shirt", "t-shirt",
            "jacket", "coat", "shoes", "sneakers", "bag",
            "handbag", "skirt", "shorts", "hat", "glasses", "watch"
        ]

        # Pre-compute CLIP text embeddings for all labels once at startup.
        # This avoids recomputing them on every request (saves ~100ms per call).
        print("Pre-computing CLIP text embeddings for categories...")
        text_inputs = self.clip_processor(text=self.labels, return_tensors="pt", padding=True)
        with torch.no_grad():
            self.text_features = self.clip_model.get_text_features(**text_inputs)
            self.text_features /= self.text_features.norm(p=2, dim=-1, keepdim=True)

        print("Locus Visual Engine Ready! (YOLO + CLIP)")

    # -------------------------------------------------------------------------
    # PUBLIC METHOD 1: detect_objects()
    # Called by the /detect endpoint.
    # Returns bounding boxes of all fashion items found in the image.
    # -------------------------------------------------------------------------
    def detect_objects(self, image_bytes):
        """
        Uses YOLOv8 to detect objects, then filters to fashion-relevant ones.

        For detected PEOPLE, it splits their bounding box into body regions
        (top/bottom/feet) so the user can select "the shirt" vs "the pants"
        independently, even when wearing a full outfit.

        Returns: list of dicts with bbox [x1,y1,x2,y2], label, and score.
        """
        t0 = time.time()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            W, H = image.size

            # Run YOLO
            # conf=0.25: minimum YOLO confidence to report a detection
            # verbose=False: suppress YOLO's own console output
            results = self.yolo(image, conf=0.25, verbose=False)[0]

            detections = []

            for box in results.boxes:
                class_id = int(box.cls[0])
                yolo_conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Case 1: YOLO found a direct fashion accessory (bag, tie...)
                if class_id in FASHION_RELEVANT_YOLO_CLASSES and class_id != 0:
                    # Use CLIP to get the fine-grained label for this crop
                    crop = image.crop((x1, y1, x2, y2))
                    clip_label, clip_conf = self._classify_crop(crop)

                    detections.append({
                        "bbox":  [x1, y1, x2, y2],
                        "label": clip_label,
                        "score": round(yolo_conf, 3)
                    })

                # Case 2: YOLO found a PERSON — split into body regions
                elif class_id == 0:
                    person_h = y2 - y1
                    person_w = x2 - x1

                    for region_name, (frac_top, frac_bot) in BODY_REGION_SPLITS.items():
                        # Calculate the sub-region bbox
                        ry1 = y1 + int(person_h * frac_top)
                        ry2 = y1 + int(person_h * frac_bot)

                        # Skip if the region is too small to be meaningful
                        region_h = ry2 - ry1
                        region_w = person_w
                        if region_h < 30 or region_w < 30:
                            continue

                        crop = image.crop((x1, ry1, x2, ry2))
                        clip_label, clip_conf = self._classify_crop(crop)

                        # Only include if CLIP is reasonably confident
                        if clip_conf >= 0.30:
                            detections.append({
                                "bbox":  [x1, ry1, x2, ry2],
                                "label": clip_label,
                                "score": round(clip_conf, 3)
                            })

            # Fallback: if YOLO found nothing, treat the whole image as one item.
            # This handles flat-lay photos (item on a table, no person) where
            # YOLO's general classes don't match fashion items well.
            if not detections:
                print("YOLO found no relevant objects. Running full-image fallback.")
                clip_label, clip_conf = self._classify_crop(image)
                if clip_conf >= 0.35:
                    detections.append({
                        "bbox":  [0, 0, W, H],
                        "label": clip_label,
                        "score": round(clip_conf, 3)
                    })

            print(f"YOLO+CLIP detected {len(detections)} items in {(time.time()-t0):.2f}s")
            return detections, W, H

        except Exception as e:
            print(f"Detection error: {e}")
            return [], 0, 0

    # -------------------------------------------------------------------------
    # PUBLIC METHOD 2: process_image()
    # Called by the /vectorize endpoint AFTER the user picks an item.
    # Takes a cropped image region -> removes background -> returns CLIP vector.
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
            # Input validation
            try:
                input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
                original_size = input_image.size
            except Exception:
                print("Not a valid image file.")
                return None, None, None

            # Resize for speed before rembg (biggest bottleneck)
            if max(input_image.size) > 512:
                input_image.thumbnail((512, 512))
                print(f"Resized {original_size} -> {input_image.size}")

            # Background removal
            print("Removing background...")
            output_image = remove(input_image, session=self.rembg_session)

            # Ghost image check:
            # If rembg removed EVERYTHING (alpha channel is all 0), reject.
            # This catches white-on-white failures and completely blank inputs.
            alpha_max = output_image.getextrema()[3][1]
            if alpha_max == 0:
                print("Ghost image: rembg removed everything. Rejecting.")
                return None, None, None

            # Smart crop to content bounding box
            bbox = output_image.getbbox()
            if bbox:
                output_image = output_image.crop(bbox)

            # Paste onto white background (CLIP expects RGB, not RGBA)
            white_bg = Image.new("RGB", output_image.size, (255, 255, 255))
            white_bg.paste(output_image, mask=output_image.split()[3])

            # CLIP vectorization
            clip_inputs = self.clip_processor(images=white_bg, return_tensors="pt")
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**clip_inputs)

            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
            vector = image_features[0].tolist()

            # Category classification with 45% confidence threshold
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
    # PRIVATE HELPER: _classify_crop()
    # Runs CLIP on a single image crop and returns (label, confidence).
    # Used internally by detect_objects() to label each YOLO bounding box.
    # -------------------------------------------------------------------------
    def _classify_crop(self, pil_image):
        """
        Runs CLIP zero-shot classification on a PIL image crop.
        Returns the best matching label from self.labels and its confidence score.
        """
        clip_inputs = self.clip_processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**clip_inputs)
        image_features /= image_features.norm(p=2, dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        top_score, top_idx = similarity[0].topk(1)

        label = self.labels[top_idx[0]]
        confidence = top_score[0].item()
        return label, confidence