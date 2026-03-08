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
    # DeepFashion2 — unchanged
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
    "top, t-shirt, sweatshirt":               "sweater",
    "sweater":                                "sweater",
    "cardigan":                               "sweater",
    "coat":                                   "coat",
    "jumpsuit":                               "jumpsuit",
    # Accessories
    "shoe":                                   "shoes",
    "sock":                                   "shoes",
    "leg warmer":                             "shoes",
    "tights, stockings":                      "shoes",
    "bag, wallet":                            "bag",
    "glasses":                                "glasses",
    "hat":                                    "hat",
    "headband, head covering, hair accessory":"hat",
    "watch":                                  "watch",
    "scarf":                                  "scarf",
}

NON_SEARCHABLE = {
    "tie", "glove", "belt", "umbrella", "cape",
    "hood", "collar", "lapel", "epaulette", "sleeve",
    "pocket", "neckline", "buckle", "zipper",
}