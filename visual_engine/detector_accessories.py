# =============================================================================
# detector_accessories.py
# Model 2: YOLOS-Fashionpedia
#
# CHANGES vs previous version:
#   - ACCESSORY_ONLY_IDS renamed to SEARCHABLE_IDS
#   - SEARCHABLE_IDS now includes clothing items DeepFashion2 can't detect:
#     sweater, cardigan, top/sweatshirt, coat, jumpsuit
#   - FASHIONPEDIA_TO_SEARCH renamed to FASHIONPEDIA_TO_CANONICAL
#     and updated to map to the new 15-label canonical vocabulary
#   - Non-searchable items (tie, glove, belt, umbrella, etc.) excluded
#   - scarf now maps to "scarf" (own category) not "bag"
# =============================================================================

import torch
from PIL import Image
from transformers import YolosForObjectDetection, YolosImageProcessor

# Single source of truth — canonical vocab and mappings live here
from clip_labels import FASHIONPEDIA_TO_CANONICAL, SEARCHABLE_IDS

# Full Fashionpedia category list — index = class_id from model output
# DO NOT reorder. These are fixed by the model weights.
FASHIONPEDIA_CATS = [
    'shirt, blouse',                          # 0
    'top, t-shirt, sweatshirt',               # 1  ← now searchable
    'sweater',                                # 2  ← now searchable
    'cardigan',                               # 3  ← now searchable
    'jacket',                                 # 4
    'vest',                                   # 5
    'pants',                                  # 6
    'shorts',                                 # 7
    'skirt',                                  # 8
    'coat',                                   # 9  ← now searchable
    'dress',                                  # 10
    'jumpsuit',                               # 11 ← now searchable
    'cape',                                   # 12 — non-searchable
    'glasses',                                # 13
    'hat',                                    # 14
    'headband, head covering, hair accessory',# 15
    'tie',                                    # 16 — non-searchable
    'glove',                                  # 17 — non-searchable
    'watch',                                  # 18
    'belt',                                   # 19 — non-searchable
    'leg warmer',                             # 20
    'tights, stockings',                      # 21
    'sock',                                   # 22
    'shoe',                                   # 23
    'bag, wallet',                            # 24
    'scarf',                                  # 25
    'umbrella',                               # 26 — non-searchable
    'hood',                                   # 27 — part of garment, not item
    'collar',                                 # 28 — part of garment
    'lapel',                                  # 29 — part of garment
    'epaulette',                              # 30 — part of garment
    'sleeve',                                 # 31 — part of garment
    'pocket',                                 # 32 — part of garment
    'neckline',                               # 33 — part of garment
    'buckle',                                 # 34 — part of garment
    'zipper',                                 # 35 — part of garment
    'applique',                               # 36 — decoration
    'bead',                                   # 37 — decoration
    'bow',                                    # 38 — decoration
    'flower',                                 # 39 — decoration
    'fringe',                                 # 40 — decoration
    'ribbon',                                 # 41 — decoration
    'rivet',                                  # 42 — decoration
    'ruffle',                                 # 43 — decoration
    'sequin',                                 # 44 — decoration
    'tassel',                                 # 45 — decoration
]

MIN_CONFIDENCE = 0.35   # lowered from 0.50 — accessories are harder to detect
MIN_AREA       = 1500   # px² — ignore tiny detections


class AccessoryDetector:
    def __init__(self):
        print("=" * 50)
        print("Loading Model 2: YOLOS-Fashionpedia")
        print("Covers: sweater/cardigan/coat/jumpsuit + accessories")
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
            classify_fn: kept for interface compatibility, not used here.
                         YOLOS already knows the category — we map it directly.

        Returns:
            list of dicts: bbox, label, search_label, score, source
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

                # Skip anything not in our searchable set
                if class_id not in SEARCHABLE_IDS:
                    continue

                x1, y1, x2, y2 = map(int, box.tolist())
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(img_w, x2); y2 = min(img_h, y2)

                if (x2 - x1) * (y2 - y1) < MIN_AREA:
                    continue

                fashionpedia_label = FASHIONPEDIA_CATS[class_id]
                canonical_label    = FASHIONPEDIA_TO_CANONICAL.get(fashionpedia_label)

                # Safety check — should never happen if SEARCHABLE_IDS and
                # FASHIONPEDIA_TO_CANONICAL are kept in sync
                if canonical_label is None:
                    print(f"  WARNING: no canonical mapping for '{fashionpedia_label}' (id {class_id}), skipping")
                    continue

                detections.append({
                    "bbox":         [x1, y1, x2, y2],
                    "label":        fashionpedia_label,  # human-readable label shown in UI
                    "search_label": canonical_label,      # canonical label used for Qdrant filter
                    "score":        round(conf, 3),
                    "source":       "yolos_fashionpedia"
                })

            print(f"  YOLOS-Fashionpedia: {len(detections)} items found")

        except Exception as e:
            print(f"  YOLOS-Fashionpedia error: {e}")

        return detections