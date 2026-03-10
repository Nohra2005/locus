# =============================================================================
# detector_accessories.py
# Model 2: YOLOS-Fashionpedia
# Detects: sweater, coat, jumpsuit (clothing DeepFashion2 misses)
#          + shoes, bag, glasses, hat, watch, scarf
#
# CHANGES vs previous version:
#   - FASHIONPEDIA_TO_CANONICAL and SEARCHABLE_IDS imported from clip_labels.py
#   - ACCESSORY_ONLY_IDS renamed to SEARCHABLE_IDS (now includes clothing items)
#   - Safety check added: warns if a searchable ID has no canonical mapping
#   - MIN_CONFIDENCE lowered to 0.35 (accessories harder to detect than clothing)
# =============================================================================

import torch
from transformers import YolosForObjectDetection, YolosImageProcessor

from clip_labels import FASHIONPEDIA_TO_CANONICAL, SEARCHABLE_IDS

# Full Fashionpedia category list — index = class_id from model output
# DO NOT reorder — fixed by model weights
FASHIONPEDIA_CATS = [
    'shirt, blouse',                           # 0
    'top, t-shirt, sweatshirt',                # 1
    'sweater',                                 # 2
    'cardigan',                                # 3
    'jacket',                                  # 4
    'vest',                                    # 5
    'pants',                                   # 6
    'shorts',                                  # 7
    'skirt',                                   # 8
    'coat',                                    # 9
    'dress',                                   # 10
    'jumpsuit',                                # 11
    'cape',                                    # 12
    'glasses',                                 # 13
    'hat',                                     # 14
    'headband, head covering, hair accessory', # 15
    'tie',                                     # 16
    'glove',                                   # 17
    'watch',                                   # 18
    'belt',                                    # 19
    'leg warmer',                              # 20
    'tights, stockings',                       # 21
    'sock',                                    # 22
    'shoe',                                    # 23
    'bag, wallet',                             # 24
    'scarf',                                   # 25
    'umbrella',                                # 26
    'hood',                                    # 27
    'collar',                                  # 28
    'lapel',                                   # 29
    'epaulette',                               # 30
    'sleeve',                                  # 31
    'pocket',                                  # 32
    'neckline',                                # 33
    'buckle',                                  # 34
    'zipper',                                  # 35
    'applique',                                # 36
    'bead',                                    # 37
    'bow',                                     # 38
    'flower',                                  # 39
    'fringe',                                  # 40
    'ribbon',                                  # 41
    'rivet',                                   # 42
    'ruffle',                                  # 43
    'sequin',                                  # 44
    'tassel',                                  # 45
]

MIN_CONFIDENCE = 0.35   # lowered from 0.50 — accessories harder to detect
MIN_AREA       = 1500   # px²


class AccessoryDetector:
    def __init__(self):
        print("=" * 50)
        print("Loading Model 2: YOLOS-Fashionpedia")
        print("Covers: sweater, coat, jumpsuit + accessories")
        print("=" * 50)
        self.processor = YolosImageProcessor.from_pretrained("valentinafeve/yolos-fashionpedia")
        self.model     = YolosForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia")
        self.model.eval()
        print("Model 2 ready.")

    def detect(self, pil_image, classify_fn):
        """
        Runs YOLOS-Fashionpedia on a PIL image.

        Args:
            pil_image:   PIL.Image — the full photo to scan
            classify_fn: kept for interface compatibility, not called here.

        Returns:
            list of dicts:
                bbox         [x1, y1, x2, y2]
                label        raw Fashionpedia label (shown in UI)
                search_label canonical label (used for Qdrant filter)
                score        YOLOS detection confidence
                source       "yolos_fashionpedia"
        """
        detections = []
        try:
            inputs       = self.processor(images=pil_image, return_tensors="pt")
            img_w, img_h = pil_image.size

            with torch.no_grad():
                outputs = self.model(**inputs)

            target_sizes = torch.tensor([[img_h, img_w]])
            results = self.processor.post_process_object_detection(
                outputs, threshold=MIN_CONFIDENCE, target_sizes=target_sizes
            )[0]

            for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
                class_id = int(label_id)
                conf     = float(score)

                if class_id not in SEARCHABLE_IDS:
                    continue

                x1, y1, x2, y2 = map(int, box.tolist())
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(img_w, x2); y2 = min(img_h, y2)

                if (x2 - x1) * (y2 - y1) < MIN_AREA:
                    continue

                fashionpedia_label = FASHIONPEDIA_CATS[class_id]
                canonical          = FASHIONPEDIA_TO_CANONICAL.get(fashionpedia_label)

                if canonical is None:
                    print(f"  WARNING: no canonical mapping for '{fashionpedia_label}' (id {class_id}), skipping")
                    continue

                detections.append({
                    "bbox":         [x1, y1, x2, y2],
                    "label":        fashionpedia_label,
                    "search_label": canonical,
                    "score":        round(conf, 3),
                    "source":       "yolos_fashionpedia"
                })

            print(f"  YOLOS-Fashionpedia: {len(detections)} items found")

        except Exception as e:
            print(f"  YOLOS-Fashionpedia error: {e}")

        return detections