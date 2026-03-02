# =============================================================================
# detector_accessories.py
# Model 2: YOLOS-Fashionpedia
# =============================================================================

import torch
from PIL import Image
from transformers import YolosForObjectDetection, YolosImageProcessor

FASHIONPEDIA_CATS = [
    'shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan',
    'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress',
    'jumpsuit', 'cape', 'glasses', 'hat',
    'headband, head covering, hair accessory', 'tie', 'glove', 'watch',
    'belt', 'leg warmer', 'tights, stockings', 'sock', 'shoe', 'bag, wallet',
    'scarf', 'umbrella', 'hood', 'collar', 'lapel', 'epaulette', 'sleeve',
    'pocket', 'neckline', 'buckle', 'zipper', 'applique', 'bead', 'bow',
    'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel',
]

ACCESSORY_ONLY_IDS = {
    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
}

# ── NEW: direct mapping instead of CLIP ───────────────────────────────────────
# CLIP gets confused when the accessory crop contains background clothing.
# YOLOS already knows the category — we just translate it to a CLIP-compatible
# label that matches what's stored in Qdrant.
FASHIONPEDIA_TO_SEARCH = {
    'glasses':                                'glasses',
    'hat':                                    'hat',
    'headband, head covering, hair accessory':'hat',
    'tie':                                    'shirt',
    'glove':                                  'bag',
    'watch':                                  'watch',
    'belt':                                   'bag',
    'leg warmer':                             'shoes',
    'tights, stockings':                      'shoes',
    'sock':                                   'shoes',
    'shoe':                                   'shoes',
    'bag, wallet':                            'bag',
    'scarf':                                  'bag',
    'umbrella':                               'bag',
}

MIN_CONFIDENCE = 0.50
MIN_AREA       = 1500


class AccessoryDetector:
    def __init__(self):
        print("=" * 50)
        print("Loading Model 2: YOLOS-Fashionpedia (Accessories)")
        print("=" * 50)
        self.processor = YolosImageProcessor.from_pretrained("valentinafeve/yolos-fashionpedia")
        self.model     = YolosForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia")
        self.model.eval()
        print("Model 2 ready.")

    def detect(self, pil_image, classify_fn):
        detections = []
        try:
            inputs   = self.processor(images=pil_image, return_tensors="pt")
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

                if class_id not in ACCESSORY_ONLY_IDS:
                    continue

                x1, y1, x2, y2 = map(int, box.tolist())
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(img_w, x2); y2 = min(img_h, y2)

                if (x2 - x1) * (y2 - y1) < MIN_AREA:
                    continue

                fashionpedia_label = FASHIONPEDIA_CATS[class_id]

                # ── CHANGED: direct mapping, no CLIP ──────────────────────────
                search_label = FASHIONPEDIA_TO_SEARCH.get(fashionpedia_label, "bag")

                detections.append({
                    "bbox":         [x1, y1, x2, y2],
                    "label":        fashionpedia_label,  # shown to user
                    "search_label": search_label,         # used for Qdrant filter
                    "score":        round(conf, 3),
                    "source":       "yolos_fashionpedia"
                })

            print(f"  YOLOS-Fashionpedia: {len(detections)} accessories found")

        except Exception as e:
            print(f"  YOLOS-Fashionpedia error: {e}")

        return detections