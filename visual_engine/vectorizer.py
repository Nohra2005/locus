# =============================================================================
# vectorizer.py
#
# ARCHITECTURE:
#
#   CATEGORY DECISION — title is ground truth, CLIP is fallback only:
#     Signal 1: title classifier (whitelist → sentence-transformers)
#     Signal 2: CLIP image classification on full image
#                ↑ only used when title gives nothing useful
#
#   WHITELIST TITLE = FINAL:
#     If the title contains an unambiguous token (whitelist hit, conf=1.0),
#     that category is returned immediately. CLIP cannot override it.
#     Reason: the image shows context (model wearing multiple items).
#     Only the title tells us what is actually for sale.
#
#   WHITELIST OVERRIDES:
#     At startup, vectorizer loads clip_labels.py (static) and then merges
#     whitelist_overrides.json on top. Approved community suggestions become
#     active without editing clip_labels.py. Reload via reload_overrides().
#
#   YOLO IS GEOMETRY ONLY — 3-tier crop selection, no full-image fallback.
#
#   DARK VECTORS: removed — never queried, wasted 50% storage.
#
# CANONICAL LABELS (13):
#   top, sports_bra, pants, leggings, shorts, skirt, dress,
#   sweater, jacket, shoes, hat, bag, jumpsuit
# =============================================================================

import copy
import json
import os
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

OVERRIDES_PATH = "/app/whitelist_overrides.json"

# =============================================================================
# CATEGORY ALIASES
# Maps voted final_category → acceptable YOLO search_label values (Tier 2).
# =============================================================================
CATEGORY_ALIASES = {
    "top":        ["top", "shirt"],
    "sports_bra": ["sports_bra", "top", "shirt"],
    "sweater":    ["sweater", "top", "shirt"],
    "jacket":     ["jacket", "coat"],
    "dress":      ["dress", "jumpsuit"],
    "jumpsuit":   ["jumpsuit", "dress"],
    "skirt":      ["skirt", "dress"],
    "pants":      ["pants", "trousers", "jeans"],
    "leggings":   ["leggings", "pants", "shorts"],
    "shorts":     ["shorts", "pants"],
    "shoes":      ["shoes", "sneakers", "boots", "sandals"],
    "hat":        ["hat", "cap"],
    "bag":        ["bag", "handbag", "backpack"],
}


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

        # Pre-compute CLIP text embeddings for canonical labels
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

        # Load token map — static whitelist + approved overrides merged on top
        self._load_token_map()

        print("=" * 50)
        print("LOCUS VISUAL ENGINE READY")
        print(f"CLIP labels: {self.clip_labels}")
        print(f"Whitelist tokens: {len(self.token_map)} "
              f"({len(UNAMBIGUOUS_TOKEN_MAP)} static + "
              f"{len(self.token_map) - len(UNAMBIGUOUS_TOKEN_MAP)} overrides)")
        print("=" * 50)

    # =========================================================================
    # PRIVATE: _load_token_map()
    # Builds self.token_map = static UNAMBIGUOUS_TOKEN_MAP + approved overrides.
    # Called at startup and again by reload_overrides().
    # =========================================================================
    def _load_token_map(self):
        self.token_map = copy.deepcopy(UNAMBIGUOUS_TOKEN_MAP)

        if not os.path.exists(OVERRIDES_PATH):
            print(f"[WHITELIST] No overrides file at {OVERRIDES_PATH} — using static map only")
            return

        try:
            with open(OVERRIDES_PATH, "r") as f:
                overrides = json.load(f)

            count = 0
            for entry in overrides:
                if entry.get("status") == "approved":
                    word     = entry.get("word", "").strip().lower()
                    category = entry.get("category", "").strip()
                    if word and category:
                        self.token_map[word] = category
                        count += 1

            print(f"[WHITELIST] Loaded {count} approved overrides from {OVERRIDES_PATH}")

        except Exception as e:
            print(f"[WHITELIST] Failed to load overrides: {e} — using static map only")

    # =========================================================================
    # PUBLIC: reload_overrides()
    # Called by the /whitelist-reload endpoint after a new override is approved.
    # Thread-safe enough for single-worker FastAPI.
    # =========================================================================
    def reload_overrides(self):
        old_count = len(self.token_map)
        self._load_token_map()
        new_count = len(self.token_map)
        print(f"[WHITELIST] Reloaded: {old_count} → {new_count} tokens")
        return {"tokens_before": old_count, "tokens_after": new_count}

    # =========================================================================
    # PUBLIC: detect_objects() — search time only
    # Returns only real YOLO boxes. No fake clip_fallback box.
    # =========================================================================
    def detect_objects(self, image_bytes):
        t0 = time.time()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            W, H  = image.size

            clothing       = self.clothing_detector.detect(image, self._classify_crop)
            accessories    = self.accessory_detector.detect(image, self._classify_crop)
            all_detections = _nms(clothing + accessories, iou_threshold=0.3)

            if not all_detections:
                print("detect_objects(): no boxes found — returning empty list")

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
            # Only meaningful when title gives nothing — the image shows context,
            # not necessarily the product being sold.
            clip_cat, clip_conf, _ = self._classify_crop(image)
            if clip_conf < 0.45:
                clip_cat = None
            print(f"[S2-CLIP]  '{title}' → '{clip_cat}' ({clip_conf:.3f})")

            # ── Vote ──────────────────────────────────────────────────────────
            final_category = self._vote(
                title_cat, title_conf, clip_cat, clip_conf, title_method
            )

            if final_category is None:
                return {"skipped": True, "skip_reason": "no_consensus"}
            if final_category == "not_fashion":
                return {"skipped": True, "skip_reason": "not_fashion"}

            print(f"[VOTE]     '{title}' → FINAL: '{final_category}'")

            # ── YOLO: find all boxes ──────────────────────────────────────────
            clothing    = self.clothing_detector.detect(image, self._classify_crop)
            accessories = self.accessory_detector.detect(image, self._classify_crop)
            detections  = _nms(clothing + accessories, iou_threshold=0.3)

            # ── 3-tier box selection — NO full-image fallback ─────────────────
            selected_box = None
            box_source   = None

            if detections:
                # Tier 1: exact label match
                exact = [d for d in detections if d.get("search_label") == final_category]
                if exact:
                    selected_box = max(exact, key=lambda d: d["score"])
                    box_source   = selected_box["source"] + "_exact"
                    print(f"[CROP T1]  Exact '{final_category}' conf={selected_box['score']:.2f}")

                # Tier 2: alias match
                if selected_box is None:
                    accepted   = CATEGORY_ALIASES.get(final_category, [final_category])
                    alias_hits = [d for d in detections if d.get("search_label") in accepted]
                    if alias_hits:
                        selected_box = max(alias_hits, key=lambda d: d["score"])
                        box_source   = selected_box["source"] + "_alias"
                        print(f"[CROP T2]  Alias '{selected_box['search_label']}' "
                              f"for '{final_category}' conf={selected_box['score']:.2f}")

                # Tier 3: best confident box, any label
                if selected_box is None:
                    confident_any = [d for d in detections if d["score"] >= 0.50]
                    if confident_any:
                        selected_box = max(confident_any, key=lambda d: d["score"])
                        box_source   = selected_box["source"] + "_best_available"
                        print(f"[CROP T3]  Best available '{selected_box['search_label']}' "
                              f"conf={selected_box['score']:.2f}")

            if selected_box is None:
                print(f"[CROP]     No usable box for '{title}' → skip")
                return {"skipped": True, "skip_reason": "no_box_found"}

            # ── Crop ──────────────────────────────────────────────────────────
            bx1, by1, bx2, by2 = selected_box["bbox"]
            bx1 = max(0, int(bx1)); by1 = max(0, int(by1))
            bx2 = min(W, int(bx2)); by2 = min(H, int(by2))
            crop = image.crop((bx1, by1, bx2, by2))

            if max(crop.size) > 512:
                crop = crop.copy()
                crop.thumbnail((512, 512))

            # ── CLIP embed ────────────────────────────────────────────────────
            vector_normal, _, _ = self._clip_embed(crop, final_category)

            print(f"[INDEX]    '{title}' done in {time.time()-t0:.2f}s  "
                  f"cat={final_category}  box={box_source}")
            return {
                "skipped":       False,
                "skip_reason":   None,
                "vector_normal": vector_normal,
                "category":      final_category,
                "box_source":    box_source,
            }

        except Exception as e:
            print(f"index_product() error for '{title}': {e}")
            return {"skipped": True, "skip_reason": str(e)}

    # =========================================================================
    # PRIVATE: _vote()
    #
    # KEY CHANGE: whitelist title hit (title_method == "whitelist") is FINAL.
    # CLIP cannot override it. The image shows context; the title says what
    # is actually for sale. A model wearing a t-shirt + leggings while the
    # product is "leggings" — CLIP sees both garments, title knows the answer.
    # =========================================================================
    def _vote(self, title_cat, title_conf, clip_cat, clip_conf, title_method=""):

        # Whitelist hit — unambiguous token, no appeal
        if title_method == "whitelist" and title_cat and title_cat != "not_fashion":
            print(f"[VOTE] Whitelist hit '{title_cat}' — final, CLIP irrelevant")
            return title_cat

        # not_fashion from title whitelist
        if title_cat == "not_fashion":
            print(f"[VOTE] Title says not_fashion → skip")
            return "not_fashion"

        # not_fashion from CLIP only when title has nothing
        if clip_cat == "not_fashion" and (title_cat is None or title_cat == "not_fashion"):
            print(f"[VOTE] CLIP says not_fashion, no title override → skip")
            return "not_fashion"

        t = title_cat if (title_cat and title_cat != "not_fashion") else None
        c = clip_cat  if (clip_cat  and clip_cat  != "not_fashion") else None

        # Both agree
        if t and c and t == c:
            print(f"[VOTE] Title + CLIP agree on '{t}'")
            return t

        # Title confident from sentence-transformers, CLIP absent
        if t and title_conf >= 0.70 and c is None:
            print(f"[VOTE] Title ST confident ({title_conf:.2f}), CLIP absent → '{t}'")
            return t

        # CLIP confident, title gave nothing at all
        if c and clip_conf >= 0.90 and t is None:
            print(f"[VOTE] CLIP confident ({clip_conf:.2f}), title absent → '{c}'")
            return c

        # Title ST confident, CLIP uncertain
        if t and title_conf >= 0.70 and c and clip_conf < 0.85:
            print(f"[VOTE] Title ST ({title_conf:.2f}) over uncertain CLIP ({clip_conf:.2f}) → '{t}'")
            return t

        # No consensus
        print(f"[VOTE] No consensus — title='{title_cat}'({title_conf:.2f}) "
              f"clip='{clip_cat}'({clip_conf:.2f}) → skip")
        return None

    # =========================================================================
    # PRIVATE: _classify_title()
    # Uses self.token_map (static whitelist + approved overrides merged).
    # =========================================================================
    def _classify_title(self, title: str):
        if not title or len(title.strip()) < 2:
            return None, "empty", 0.0

        tokens = title.lower().replace("-", " ").replace("/", " ").split()

        fashion_hits     = {}
        not_fashion_hits = 0

        for token in tokens:
            cat = self.token_map.get(token)
            if cat is None:
                continue
            if cat == "not_fashion":
                not_fashion_hits += 1
            else:
                fashion_hits[cat] = fashion_hits.get(cat, 0) + 1

        bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
        for bigram in bigrams:
            cat = self.token_map.get(bigram)
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

        # Sentence-transformers fallback
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
    # PRIVATE: _clip_embed() — unchanged
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
    # PRIVATE: _classify_crop() — unchanged
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