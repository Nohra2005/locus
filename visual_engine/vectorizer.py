# =============================================================================
# vectorizer.py
#
# ARCHITECTURE:
#
#   CATEGORY DECISION — 2 signals only:
#     Signal 1: title classifier (whitelist → sentence-transformers)
#     Signal 2: CLIP image classification on full image
#
#   YOLO IS GEOMETRY ONLY — its labels are never used for category decision.
#   After category is decided, YOLO boxes are re-ranked by CLIP similarity
#   to the decided category. Best-scoring crop wins.
#
#   BOUNDING BOX SELECTION:
#     1. YOLO finds all boxes in the image
#     2. For each box: crop → CLIP cosine similarity vs final_category text
#     3. Pick box with highest CLIP-category similarity score
#     4. If no boxes or best score too low → use full image
#
# VOTING (title + CLIP only):
#   1. Both agree                        → winner
#   2. title confident (>= 0.70), CLIP absent/low → title wins
#   3. CLIP very confident (>= 0.90), title absent/wrong → CLIP wins
#   4. Both disagree, neither confident  → skip
# =============================================================================

import torch
import io
import base64
import time
from PIL import Image, ImageEnhance
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer, util

from detector_clothing import ClothingDetector
from detector_accessories import AccessoryDetector
from clip_labels import (
    CANONICAL_LABELS,
    LABEL_DESCRIPTIONS,
    UNAMBIGUOUS_TOKEN_MAP,
)

# Minimum CLIP-vs-category similarity for a box to be used as crop.
# Below this threshold the crop doesn't visually match the category
# well enough — fall back to full image.



# =============================================================================
# NMS HELPERS
# =============================================================================

def _iou(a, b):
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


def _nms(detections, iou_threshold=0.3):
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

        self.clothing_detector  = ClothingDetector()
        self.accessory_detector = AccessoryDetector()

        print("Loading CLIP ViT-B/16...")
        self.clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.clip_labels    = CANONICAL_LABELS

        # Pre-compute CLIP text embeddings for the 15 canonical labels
        text_inputs = self.clip_processor(
            text=self.clip_labels, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            self.text_features = self.clip_model.get_text_features(**text_inputs)
            self.text_features /= self.text_features.norm(p=2, dim=-1, keepdim=True)


        print("Loading sentence-transformers (all-MiniLM-L6-v2)...")
        self.st_model         = SentenceTransformer("all-MiniLM-L6-v2")
        self._all_categories  = list(LABEL_DESCRIPTIONS.keys())
        desc_texts            = [LABEL_DESCRIPTIONS[c] for c in self._all_categories]
        self._desc_embeddings = self.st_model.encode(
            desc_texts, convert_to_tensor=True, normalize_embeddings=True
        )

        print("=" * 50)
        print("LOCUS VISUAL ENGINE READY")
        print(f"CLIP labels: {self.clip_labels}")
        print(f"Whitelist tokens: {len(UNAMBIGUOUS_TOKEN_MAP)}")
        print("=" * 50)

    # =========================================================================
    # PUBLIC: detect_objects() — search time only
    # =========================================================================
    def detect_objects(self, image_bytes):
        t0 = time.time()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            W, H  = image.size

            clothing    = self.clothing_detector.detect(image, self._classify_crop)
            accessories = self.accessory_detector.detect(image, self._classify_crop)
            all_detections = _nms(clothing + accessories, iou_threshold=0.3)

            if not all_detections:
                label, conf, _ = self._classify_crop(image)
                if conf >= 0.35:
                    all_detections.append({
                        "bbox":         [0, 0, W, H],
                        "label":        label,
                        "search_label": label,
                        "score":        round(conf, 3),
                        "source":       "clip_fallback",
                    })

            print(f"detect_objects(): {len(all_detections)} boxes in {time.time()-t0:.2f}s")
            return all_detections, W, H

        except Exception as e:
            print(f"detect_objects() error: {e}")
            return [], 0, 0

    # =========================================================================
    # PUBLIC: process_image() — search time, pre-cropped bytes
    # =========================================================================
    def process_image(self, image_bytes, yolo_label="", darken=False):
        t0 = time.time()
        try:
            input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            if max(input_image.size) > 512:
                input_image.thumbnail((512, 512))

            clip_input    = ImageEnhance.Brightness(input_image).enhance(0.3) if darken else input_image
            vector, category_tag, category_scores = self._clip_embed(clip_input, yolo_label)

            buf = io.BytesIO()
            input_image.save(buf, format="PNG")
            debug_img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            print(f"process_image(darken={darken}) done in {time.time()-t0:.2f}s")
            return vector, category_tag, category_scores, debug_img_b64

        except Exception as e:
            print(f"process_image() error: {e}")
            return None, None, None, None

    # =========================================================================
    # PUBLIC: index_product() — index time only
    #
    # Flow:
    #   1. Classify title → title_cat
    #   2. CLIP on full image → clip_cat
    #   3. Vote (title + CLIP only) → final_category
    #   4. YOLO detects all boxes (labels ignored)
    #   5. For each box: CLIP-vs-category score
    #   6. Best scoring box above threshold → crop to it
    #      else → full image
    #   7. CLIP embed crop (normal + dark) → vectors
    # =========================================================================
    def index_product(self, image_bytes: bytes, title: str = ""):
        t0 = time.time()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            W, H  = image.size

            # ── Signal 1: title ───────────────────────────────────────────────
            title_cat, title_method, title_conf = self._classify_title(title)
            print(f"[S1-TITLE] '{title}' → '{title_cat}' via {title_method} ({title_conf:.3f})")

            # ── Signal 2: CLIP on full image ──────────────────────────────────
            clip_cat, clip_conf, _ = self._classify_crop(image)
            if clip_conf < 0.45:
                clip_cat = None
            print(f"[S2-CLIP]  '{title}' → '{clip_cat}' ({clip_conf:.3f})")

            # ── Vote (title + CLIP only) ──────────────────────────────────────
            final_category = self._vote(title_cat, title_conf, clip_cat, clip_conf)

            if final_category is None:
                return {"skipped": True, "skip_reason": "no_consensus"}
            if final_category == "not_fashion":
                return {"skipped": True, "skip_reason": "not_fashion"}

            print(f"[VOTE]     '{title}' → FINAL: '{final_category}'")

            # ── YOLO: find all boxes (labels ignored) ─────────────────────────
            clothing    = self.clothing_detector.detect(image, self._classify_crop)
            accessories = self.accessory_detector.detect(image, self._classify_crop)
            detections  = _nms(clothing + accessories, iou_threshold=0.3)

            # ── Select best crop: YOLO box matching voted category ────────────
            # YOLO's geometry is reliable — pick the highest-confidence box
            # whose label matches final_category. No extra CLIP passes needed.
            crop       = image
            box_source = "full_image"

            if detections:
                matching = [d for d in detections if d.get("search_label") == final_category]
                if matching:
                    best_match         = max(matching, key=lambda d: d["score"])
                    bx1, by1, bx2, by2 = best_match["bbox"]
                    bx1 = max(0, int(bx1)); by1 = max(0, int(by1))
                    bx2 = min(W, int(bx2)); by2 = min(H, int(by2))
                    crop               = image.crop((bx1, by1, bx2, by2))
                    box_source         = best_match["source"]
                    print(f"[CROP]     Matched '{final_category}' box: {box_source} yolo_conf={best_match['score']:.2f}")
                else:
                    # No box matches the category — use full image
                    # Category is already trusted (voted) so full image is safe
                    print(f"[CROP]     No '{final_category}' box in detections → full image")
            else:
                print(f"[CROP]     No YOLO detections → full image")

            # ── Resize ────────────────────────────────────────────────────────
            if max(crop.size) > 512:
                crop = crop.copy()
                crop.thumbnail((512, 512))

            # ── CLIP embed normal + dark ──────────────────────────────────────
            vector_normal, _, _ = self._clip_embed(crop, final_category)
            dark_crop           = ImageEnhance.Brightness(crop).enhance(0.3)
            vector_dark,   _, _ = self._clip_embed(dark_crop, final_category)

            print(f"[INDEX]    '{title}' done in {time.time()-t0:.2f}s  "
                  f"cat={final_category}  box={box_source}")
            return {
                "skipped":       False,
                "skip_reason":   None,
                "vector_normal": vector_normal,
                "vector_dark":   vector_dark,
                "category":      final_category,
                "box_source":    box_source,
            }

        except Exception as e:
            print(f"index_product() error for '{title}': {e}")
            return {"skipped": True, "skip_reason": str(e)}

    # =========================================================================
    # PRIVATE: _vote()
    # Title + CLIP only. YOLO label is never used here.
    #
    # Rules:
    #   1. not_fashion from either signal (if one says not_fashion confidently) → skip
    #   2. Both agree on same fashion category → winner
    #   3. title confident (>= 0.70), CLIP absent/low → title wins
    #   4. CLIP confident (>= 0.90), title absent/wrong → CLIP wins
    #   5. no consensus → skip
    # =========================================================================
    def _vote(self, title_cat, title_conf, clip_cat, clip_conf):

        # Rule 1: not_fashion
        # If title says not_fashion confidently → skip immediately
        if title_cat == "not_fashion":
            print(f"[VOTE] Title says not_fashion → skip")
            return "not_fashion"
        # If CLIP says not_fashion very confidently and title has no category → skip
        if clip_cat == "not_fashion" and (title_cat is None or title_cat == "not_fashion"):
            print(f"[VOTE] CLIP says not_fashion, no title override → skip")
            return "not_fashion"

        # Only work with fashion categories from here
        t = title_cat if (title_cat and title_cat != "not_fashion") else None
        c = clip_cat  if (clip_cat  and clip_cat  != "not_fashion") else None

        # Rule 2: both agree
        if t and c and t == c:
            print(f"[VOTE] Title + CLIP agree on '{t}'")
            return t

        # Rule 3: title confident, CLIP absent or low confidence
        if t and title_conf >= 0.70 and c is None:
            print(f"[VOTE] Title confident ({title_conf:.2f}), CLIP absent → '{t}'")
            return t

        # Rule 4: CLIP confident, title absent or wrong
        if c and clip_conf >= 0.90 and t is None:
            print(f"[VOTE] CLIP confident ({clip_conf:.2f}), title absent → '{c}'")
            return c

        # Rule 4b: CLIP very confident, overrides disagreeing title
        if c and clip_conf >= 0.95 and t and t != c:
            print(f"[VOTE] CLIP very confident ({clip_conf:.2f}) overrides title '{t}' → '{c}'")
            return c

        # Rule 5: title present, CLIP present but they disagree moderately
        # Trust title if it came from whitelist (conf=1.0) and CLIP isn't dominant
        if t and title_conf >= 0.70 and c and clip_conf < 0.85:
            print(f"[VOTE] Title whitelist ({title_conf:.2f}) over uncertain CLIP ({clip_conf:.2f}) → '{t}'")
            return t

        # Rule 6: no consensus
        print(f"[VOTE] No consensus — title='{title_cat}'({title_conf:.2f}) "
              f"clip='{clip_cat}'({clip_conf:.2f}) → skip")
        return None

    # =========================================================================
    # PRIVATE: _classify_title() — whitelist → sentence-transformers
    # =========================================================================
    def _classify_title(self, title: str):
        if not title or len(title.strip()) < 2:
            return None, "empty", 0.0

        tokens = title.lower().replace("-", " ").replace("/", " ").split()

        # Layer 1: whitelist
        fashion_hits     = {}
        not_fashion_hits = 0

        for token in tokens:
            cat = UNAMBIGUOUS_TOKEN_MAP.get(token)
            if cat is None:
                continue
            if cat == "not_fashion":
                not_fashion_hits += 1
            else:
                fashion_hits[cat] = fashion_hits.get(cat, 0) + 1

        bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
        for bigram in bigrams:
            cat = UNAMBIGUOUS_TOKEN_MAP.get(bigram)
            if cat is None:
                continue
            if cat == "not_fashion":
                not_fashion_hits += 2
            else:
                fashion_hits[cat] = fashion_hits.get(cat, 0) + 2

        if fashion_hits:
            best_cat   = max(fashion_hits, key=fashion_hits.get)
            best_count = fashion_hits[best_cat]
            if not_fashion_hits >= best_count:
                return "not_fashion", "whitelist", 1.0
            return best_cat, "whitelist", 1.0

        if not_fashion_hits > 0:
            return "not_fashion", "whitelist", 1.0

        # Layer 2: sentence-transformers
        title_emb = self.st_model.encode(
            title, convert_to_tensor=True, normalize_embeddings=True
        )
        scores = util.cos_sim(title_emb, self._desc_embeddings)[0]

        not_fashion_idx   = self._all_categories.index("not_fashion")
        not_fashion_score = float(scores[not_fashion_idx])

        best_fashion_score = -1.0
        best_fashion_cat   = None
        for i, cat in enumerate(self._all_categories):
            if cat == "not_fashion":
                continue
            s = float(scores[i])
            if s > best_fashion_score:
                best_fashion_score = s
                best_fashion_cat   = cat

        if not_fashion_score >= 0.40 and not_fashion_score >= best_fashion_score:
            return "not_fashion", "sentence_transformer", not_fashion_score

        if best_fashion_score >= 0.30:
            return best_fashion_cat, "sentence_transformer", best_fashion_score

        return None, "none", 0.0

    # =========================================================================
    # PRIVATE: _clip_embed() — full CLIP forward pass for vectorization
    # =========================================================================
    def _clip_embed(self, pil_image: Image.Image, label_hint: str = ""):
        clip_inputs = self.clip_processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**clip_inputs)
        image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
        vector = image_features[0].tolist()

        similarity    = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        scores_tensor = similarity[0]

        category_scores = {
            label: round(scores_tensor[i].item(), 4)
            for i, label in enumerate(self.clip_labels)
        }

        if label_hint and label_hint.strip() in self.clip_labels:
            category_tag = label_hint.strip()
        else:
            clip_conf    = scores_tensor.max().item()
            clip_label   = self.clip_labels[scores_tensor.argmax().item()]
            category_tag = clip_label if clip_conf >= 0.45 else None

        return vector, category_tag, category_scores

    # =========================================================================
    # PRIVATE: _classify_crop() — used by YOLO detectors + Signal 2
    # =========================================================================
    def _classify_crop(self, pil_image):
        _, label, scores = self._clip_embed(pil_image)
        best_conf = max(scores.values()) if scores else 0.0
        return label or "", best_conf, scores

    # =========================================================================
    # PUBLIC: classify_text() — debug endpoint
    # =========================================================================
    def classify_text(self, title: str):
        category, method, confidence = self._classify_title(title)
        return category, confidence