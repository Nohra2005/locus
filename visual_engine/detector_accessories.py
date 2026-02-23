# =============================================================================
# detector_accessories.py
# Model 2: YOLOv8n COCO
# Detects accessories and footwear: shoes, bags, ties, suitcases
# Does NOT detect clothing items (shirts, pants, dresses, etc.)
# =============================================================================

from ultralytics import YOLO
from PIL import Image

# COCO class IDs we care about — everything else is ignored
COCO_FASHION_CLASSES = {
    24: "handbag",
    25: "handbag",   # umbrella — sometimes catches tote bags
    27: "tie",
    28: "suitcase",
    66: "shoes",     # sneaker in COCO
}

MIN_CONFIDENCE = 0.30
MIN_AREA = 1500  # px²


class AccessoryDetector:
    def __init__(self):
        print("=" * 50)
        print("Loading Model 2: YOLOv8 COCO (Shoes & Bags)")
        print("Covers: shoes, handbags, ties, suitcases")
        print("=" * 50)
        # Downloads yolov8n.pt automatically (~6MB) on first run
        self.model = YOLO("yolov8n.pt")
        print("Model 2 ready.")

    def detect(self, pil_image, classify_fn):
        """
        Runs YOLOv8 COCO on a PIL image, filtering to fashion accessories only.

        Args:
            pil_image:   PIL.Image — the full photo to scan
            classify_fn: function(pil_image) -> (label, confidence)
                         Provided by the orchestrator (CLIP).
                         Used to refine the coarse COCO label.

        Returns:
            list of dicts: bbox, label, search_label, score, source
        """
        detections = []
        try:
            results = self.model(pil_image, conf=MIN_CONFIDENCE, verbose=False)[0]

            for box in results.boxes:
                class_id = int(box.cls[0])

                # Ignore everything that isn't a fashion accessory
                if class_id not in COCO_FASHION_CLASSES:
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Skip boxes that are too small to be meaningful
                if (x2 - x1) * (y2 - y1) < MIN_AREA:
                    continue

                coco_label = COCO_FASHION_CLASSES[class_id]

                # Use CLIP (passed in as classify_fn) to refine the label
                crop = pil_image.crop((x1, y1, x2, y2))
                clip_label, _ = classify_fn(crop)

                detections.append({
                    "bbox":         [x1, y1, x2, y2],
                    "label":        coco_label,  # shown to user: "shoes" / "handbag"
                    "search_label": clip_label,  # used for Qdrant filter
                    "score":        round(conf, 3),
                    "source":       "yolo_coco"
                })

            print(f"  YOLOv8 COCO: {len(detections)} accessories found")

        except Exception as e:
            print(f"  YOLOv8 COCO error: {e}")

        return detections
