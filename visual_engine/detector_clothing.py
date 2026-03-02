# =============================================================================
# detector_clothing.py
# Model 1: DeepFashion2 YOLOv8
# Detects clothing items: shirts, pants, dresses, skirts, outwear
# Does NOT detect shoes, bags, or accessories
# =============================================================================

from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image

DEEPFASHION2_LABELS = {
    0:  "short sleeved shirt",
    1:  "long sleeved shirt",
    2:  "short sleeved outwear",
    3:  "long sleeved outwear",
    4:  "vest",
    5:  "sling",
    6:  "shorts",
    7:  "trousers",
    8:  "skirt",
    9:  "short sleeved dress",
    10: "long sleeved dress",
    11: "vest dress",
    12: "sling dress",
}

MIN_CONFIDENCE = 0.30
MIN_AREA = 1500  # px²


class ClothingDetector:
    def __init__(self):
        print("=" * 50)
        print("Loading Model 1: DeepFashion2 (Clothing)")
        print("Covers: shirts, pants, dresses, skirts, outwear")
        print("=" * 50)
        model_path = hf_hub_download(
            repo_id="Bingsu/adetailer",
            filename="deepfashion2_yolov8s-seg.pt"
        )
        self.model = YOLO(model_path)
        print("Model 1 ready.")

    def detect(self, pil_image, classify_fn):
        """
        Runs DeepFashion2 on a PIL image.

        Args:
            pil_image:   PIL.Image — the full photo to scan
            classify_fn: function(pil_image) -> (label, confidence)
                         Provided by the orchestrator (CLIP).
                         Used to map DeepFashion2 labels to search-friendly labels.

        Returns:
            list of dicts: bbox, label, search_label, score, source
        """
        detections = []
        try:
            results = self.model(pil_image, conf=MIN_CONFIDENCE, verbose=False)[0]

            for box in results.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Skip boxes that are too small to be meaningful
                if (x2 - x1) * (y2 - y1) < MIN_AREA:
                    continue

                df2_label = DEEPFASHION2_LABELS.get(class_id, "clothing")

                # Use CLIP (passed in as classify_fn) to get a consistent
                # search label that matches the Qdrant index categories
                crop = pil_image.crop((x1, y1, x2, y2))
                clip_label, _ = classify_fn(crop)

                detections.append({
                    "bbox":         [x1, y1, x2, y2],
                    "label":        df2_label,   # shown to user
                    "search_label": clip_label,  # used for Qdrant filter
                    "score":        round(conf, 3),
                    "source":       "deepfashion2"
                })

            print(f"  DeepFashion2: {len(detections)} clothing items found")

        except Exception as e:
            print(f"  DeepFashion2 error: {e}")

        return detections
