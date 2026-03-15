# =============================================================================
# clip_labels.py — Single source of truth for all category labels
#
# Imported by: vectorizer.py, detector_clothing.py, detector_accessories.py
# Never hardcode labels anywhere else — always import from here.
# =============================================================================

CANONICAL_LABELS = [
    "shirt",
    "sweater",      # hoodie, cardigan, knitwear, sweatshirt
    "jacket",
    "coat",         # trench, puffer, overcoat — longer/heavier than jacket
    "dress",
    "jumpsuit",     # romper, playsuit, overalls — one-piece
    "skirt",
    "pants",
    "shorts",
    "shoes",
    "bag",
    "glasses",
    "hat",
    "watch",
    "scarf",
]  # 15 labels total

YOLO_TO_CANONICAL = {
    # DeepFashion2 class names → canonical labels
    "short sleeved shirt":   "shirt",
    "long sleeved shirt":    "shirt",
    "short sleeved outwear": "jacket",
    "long sleeved outwear":  "jacket",
    "vest":                  "jacket",
    "sling":                 "shirt",
    "shorts":                "shorts",
    "trousers":              "pants",
    "skirt":                 "skirt",
    "short sleeved dress":   "dress",
    "long sleeved dress":    "dress",
    "vest dress":            "dress",
    "sling dress":           "dress",
}

FASHIONPEDIA_TO_CANONICAL = {
    # Clothing items DeepFashion2 misses — handled by YOLOS
    "top, t-shirt, sweatshirt":                "shirt",
    "sweater":                                 "sweater",
    "cardigan":                                "sweater",
    "coat":                                    "coat",
    "jumpsuit":                                "jumpsuit",
    # Accessories
    "shoe":                                    "shoes",
    "sock":                                    "shoes",
    "leg warmer":                              "shoes",
    "tights, stockings":                       "shoes",
    "bag, wallet":                             "bag",
    "glasses":                                 "glasses",
    "hat":                                     "hat",
    "headband, head covering, hair accessory": "hat",
    "watch":                                   "watch",
    "scarf":                                   "scarf",
}

NON_SEARCHABLE = {
    "tie", "glove", "belt", "umbrella", "cape",
    "hood", "collar", "lapel", "epaulette", "sleeve",
    "pocket", "neckline", "buckle", "zipper",
}

# Fashionpedia class IDs we actually want to detect.
# Every ID whose Fashionpedia label has a mapping in FASHIONPEDIA_TO_CANONICAL.
# IDs NOT in this set (tie, glove, belt, garment parts...) are silently skipped.
SEARCHABLE_IDS = {
    1,   # top, t-shirt, sweatshirt  → shirt
    2,   # sweater                   → sweater
    3,   # cardigan                  → sweater
    9,   # coat                      → coat
    11,  # jumpsuit                  → jumpsuit
    13,  # glasses                   → glasses
    14,  # hat                       → hat
    15,  # headband / hair accessory → hat
    18,  # watch                     → watch
    20,  # leg warmer                → shoes
    21,  # tights, stockings         → shoes
    22,  # sock                      → shoes
    23,  # shoe                      → shoes
    24,  # bag, wallet               → bag
    25,  # scarf                     → scarf
}