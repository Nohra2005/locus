# =============================================================================
# clip_labels.py
# Single source of truth for Locus category vocabulary.
#
# Imported by:
#   - vectorizer.py           (CANONICAL_LABELS for CLIP classification)
#   - detector_clothing.py    (YOLO_TO_CANONICAL for DeepFashion2 mapping)
#   - detector_accessories.py (FASHIONPEDIA_TO_CANONICAL + SEARCHABLE_IDS)
#
# To add a new category: add it to CANONICAL_LABELS, add its mappings below,
# add its Fashionpedia class IDs to SEARCHABLE_IDS. Restart Docker. Done.
# =============================================================================

# 15 canonical labels — what CLIP classifies against, what Qdrant filters on
CANONICAL_LABELS = [
    "shirt",       # t-shirt, blouse, sling top
    "sweater",     # hoodie, cardigan, knitwear
    "jacket",      # outwear, vest, blazer
    "coat",        # trench, puffer, overcoat
    "dress",       # all dress variants
    "jumpsuit",    # romper, playsuit, overalls
    "skirt",
    "pants",       # trousers, leggings
    "shorts",
    "shoes",       # sneakers, boots
    "bag",         # handbag, wallet
    "glasses",     # sunglasses
    "hat",         # headband, hair accessory
    "watch",
    "scarf",
]

# =============================================================================
# YOLO_TO_CANONICAL
# DeepFashion2 YOLOv8 raw labels → canonical labels.
# Used by detector_clothing.py.
# DeepFashion2 covers: shirt, jacket, dress, skirt, pants, shorts.
# Does NOT cover: sweater, coat, jumpsuit — those come from Fashionpedia.
# =============================================================================
YOLO_TO_CANONICAL = {
    "short sleeved shirt":   "shirt",
    "long sleeved shirt":    "shirt",
    "sling":                 "shirt",
    "short sleeved outwear": "jacket",
    "long sleeved outwear":  "jacket",
    "vest":                  "jacket",
    "shorts":                "shorts",
    "trousers":              "pants",
    "skirt":                 "skirt",
    "short sleeved dress":   "dress",
    "long sleeved dress":    "dress",
    "vest dress":            "dress",
    "sling dress":           "dress",
}

# =============================================================================
# FASHIONPEDIA_TO_CANONICAL
# YOLOS-Fashionpedia raw labels → canonical labels.
# Used by detector_accessories.py.
#
# Decisions:
#   "top, t-shirt, sweatshirt" → "shirt"   (NOT sweater — t-shirts dominate)
#   "leg warmer"               → removed   (niche, maps poorly, CLIP fallback handles)
#   "tights, stockings"        → removed   (niche, maps poorly, CLIP fallback handles)
#   "sock"                     → "shoes"   (footwear family, acceptable)
# =============================================================================
FASHIONPEDIA_TO_CANONICAL = {
    "top, t-shirt, sweatshirt":                "shirt",
    "sweater":                                 "sweater",
    "cardigan":                                "sweater",
    "coat":                                    "coat",
    "jumpsuit":                                "jumpsuit",
    "shoe":                                    "shoes",
    "sock":                                    "shoes",
    "bag, wallet":                             "bag",
    "glasses":                                 "glasses",
    "hat":                                     "hat",
    "headband, head covering, hair accessory": "hat",
    "watch":                                   "watch",
    "scarf":                                   "scarf",
}

# =============================================================================
# SEARCHABLE_IDS
# Fashionpedia class IDs to keep. Everything else is discarded.
#
# Removed:
#   ID 1  — "top, t-shirt, sweatshirt": DeepFashion2 handles shirts better,
#            keeping this causes duplicate bounding boxes on shirts
#   ID 20 — "leg warmer": niche item, removed from searchable
#   ID 21 — "tights, stockings": niche item, removed from searchable
# =============================================================================
SEARCHABLE_IDS = {
    # Clothing DeepFashion2 cannot detect
    2,   # sweater                    → sweater
    3,   # cardigan                   → sweater
    9,   # coat                       → coat
    11,  # jumpsuit                   → jumpsuit

    # Accessories
    13,  # glasses                    → glasses
    14,  # hat                        → hat
    15,  # headband / hair accessory  → hat
    18,  # watch                      → watch
    22,  # sock                       → shoes
    23,  # shoe                       → shoes
    24,  # bag, wallet                → bag
    25,  # scarf                      → scarf
}

# =============================================================================
# NON_SEARCHABLE (documentation only — not imported anywhere)
# Fashionpedia items intentionally excluded and why.
# =============================================================================
# ID 0  "shirt, blouse"         — DeepFashion2 handles this better
# ID 1  "top, t-shirt, ..."     — DeepFashion2 handles this, avoids duplicates
# ID 4  "jacket"                — DeepFashion2 handles this better
# ID 5  "vest"                  — DeepFashion2 handles this better
# ID 6  "pants"                 — DeepFashion2 handles this better
# ID 7  "shorts"                — DeepFashion2 handles this better
# ID 8  "skirt"                 — DeepFashion2 handles this better
# ID 10 "dress"                 — DeepFashion2 handles this better
# ID 12 "cape"                  — too rare, maps ambiguously
# ID 16 "tie"                   — not searchable in Locus
# ID 17 "glove"                 — not searchable in Locus
# ID 19 "belt"                  — not searchable in Locus
# ID 20 "leg warmer"            — niche, maps poorly
# ID 21 "tights, stockings"     — niche, maps poorly
# ID 26 "umbrella"              — not a fashion item
# ID 27+ garment parts          — buckle, zipper, sequin, etc.