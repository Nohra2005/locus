# =============================================================================
# detector_accessories.py
# Model 2: YOLOS-Fashionpedia
#
# Replaces the broken YOLOv8 COCO model which only knew "handbag" and "sneaker".
#
# YOLOS-Fashionpedia was fine-tuned on 46,781 fashion images and detects
# 46 categories including all the accessories we need:
#   glasses · hat · watch · belt · shoe · bag/wallet
#   scarf · glove · sock · headband · tie · umbrella
#   + all clothing categories (handled by DeepFashion2 but overlaps are fine)
#
# Architecture: YOLOS (Vision Transformer), NOT YOLOv8.
# Uses HuggingFace transformers library instead of ultralytics.
# =============================================================================

import torch
from PIL import Image
from transformers import YolosForObjectDetection, YolosImageProcessor

# Full list of 46 Fashionpedia categories (index = class id)
FASHIONPEDIA_CATS = [
    'shirt, blouse',
    'top, t-shirt, sweatshirt',
    'sweater',
    'cardigan',
    'jacket',
    'vest',
    'pants',
    'shorts',
    'skirt',
    'coat',
    'dress',
    'jumpsuit',
    'cape',
    'glasses',
    'hat',
    'headband, head covering, hair accessory',
    'tie',
    'glove',
    'watch',
    'belt',
    'leg warmer',
    'tights, stockings',
    'sock',
    'shoe',
    'bag, wallet',
    'scarf',
    'umbrella',
    'hood',
    'collar',
    'lapel',
    'epaulette',
    'sleeve',
    'pocket',
    'neckline',
    'buckle',
    'zipper',
    'applique',
    'bead',
    'bow',
    'flower',
    'fringe',
    'ribbon',
    'rivet',
    'ruffle',
    'sequin',
    'tassel',
]

# We only care about accessories — DeepFashion2 already handles clothing.
# This set filters out clothing categories to avoid showing duplicates
# to the user (e.g. both models detecting the same shirt).
ACCESSORY_ONLY_IDS = {
    13,  # glasses
    14,  # hat
    15,  # headband, head covering, hair accessory
    16,  # tie
    17,  # glove
    18,  # watch
    19,  # belt
    20,  # leg warmer
    21,  # tights, stockings
    22,  # sock
    23,  # shoe
    24,  # bag, wallet
    25,  # scarf
    26,  # umbrella
}

MIN_CONFIDENCE = 0.50   # YOLOS tends to be verbose — higher threshold reduces noise
MIN_AREA       = 1500   # px² — ignore tiny false positives


class AccessoryDetector:
    def __init__(self):
        print("=" * 50)
        print("Loading Model 2: YOLOS-Fashionpedia (Accessories)")
        print("Covers: glasses, hat, watch, belt, shoe, bag,")
        print("        scarf, glove, sock, tie, headband, umbrella")
        print("Model size: ~123MB (downloads once, then cached)")
        print("=" * 50)

        # Downloads ~123MB on first run, cached in HuggingFace cache after that
        self.processor = YolosImageProcessor.from_pretrained(
            "valentinafeve/yolos-fashionpedia"
        )
        self.model = YolosForObjectDetection.from_pretrained(
            "valentinafeve/yolos-fashionpedia"
        )
        self.model.eval()   # Set to inference mode (disables dropout etc.)
        print("Model 2 ready.")

    def detect(self, pil_image, classify_fn):
        """
        Runs YOLOS-Fashionpedia on a PIL image.
        Only returns accessory detections — clothing is handled by DeepFashion2.

        Args:
            pil_image:   PIL.Image — the full photo to scan
            classify_fn: function(pil_image) -> (label, confidence)
                         CLIP function passed in from the orchestrator.
                         Used to get a consistent search label from the
                         raw Fashionpedia label.

        Returns:
            list of dicts: bbox, label, search_label, score, source
        """
        detections = []
        try:
            # ── Preprocess ───────────────────────────────────────────────────
            # YOLOS needs the image in a specific format.
            # The processor handles resizing and normalization automatically.
            inputs = self.processor(images=pil_image, return_tensors="pt")
            img_w, img_h = pil_image.size

            # ── Inference ────────────────────────────────────────────────────
            with torch.no_grad():
                outputs = self.model(**inputs)

            # ── Post-process: convert raw outputs to bounding boxes ───────────
            # target_sizes tells the processor the original image dimensions
            # so it can scale the boxes back to pixel coordinates.
            target_sizes = torch.tensor([[img_h, img_w]])
            results = self.processor.post_process_object_detection(
                outputs,
                threshold=MIN_CONFIDENCE,
                target_sizes=target_sizes
            )[0]

            # ── Filter and format results ─────────────────────────────────────
            for score, label_id, box in zip(
                results["scores"],
                results["labels"],
                results["boxes"]
            ):
                class_id = int(label_id)
                conf     = float(score)

                # Only keep accessories — skip clothing (DeepFashion2 handles those)
                if class_id not in ACCESSORY_ONLY_IDS:
                    continue

                # Convert box from [cx, cy, w, h] → [x1, y1, x2, y2] pixel coords
                x1, y1, x2, y2 = map(int, box.tolist())

                # Clamp to image bounds (YOLOS occasionally predicts outside)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(img_w, x2); y2 = min(img_h, y2)

                # Skip boxes that are too small
                if (x2 - x1) * (y2 - y1) < MIN_AREA:
                    continue

                # Get the human-readable Fashionpedia label
                fashionpedia_label = FASHIONPEDIA_CATS[class_id]

                # Use CLIP to get a consistent search label that matches
                # what's indexed in Qdrant (e.g. "bag, wallet" → "bag")
                crop       = pil_image.crop((x1, y1, x2, y2))
                clip_label, _ = classify_fn(crop)

                detections.append({
                    "bbox":         [x1, y1, x2, y2],
                    "label":        fashionpedia_label,  # shown to user: "bag, wallet"
                    "search_label": clip_label,          # used for Qdrant: "bag"
                    "score":        round(conf, 3),
                    "source":       "yolos_fashionpedia"
                })

            print(f"  YOLOS-Fashionpedia: {len(detections)} accessories found")

        except Exception as e:
            print(f"  YOLOS-Fashionpedia error: {e}")

        return detections