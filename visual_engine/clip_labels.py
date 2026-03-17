# =============================================================================
# clip_labels.py — Single source of truth for all category labels
# =============================================================================

CANONICAL_LABELS = [
    "shirt",
    "sweater",
    "jacket",
    "coat",
    "dress",
    "jumpsuit",
    "skirt",
    "pants",
    "shorts",
    "shoes",
    "bag",
    "glasses",
    "hat",
    "watch",
    "scarf",
]

# =============================================================================
# UNAMBIGUOUS_TOKEN_MAP
#
# Whitelist of single tokens that map to exactly one category regardless
# of context. The test: "can this word alone ever correctly mean a different
# category?" If yes — it is NOT here. Sentence-transformers handles it.
#
# Rules:
#   - No brand names (brands are not categories)
#   - No modifier words (hiking, ski, compression, training, running...)
#   - No fabric/material words (wool, cotton, linen...)
#   - No style words (classic, slim, wide, oversized...)
#   - Only the word alone = the category, unambiguously, always
# =============================================================================

UNAMBIGUOUS_TOKEN_MAP = {

    # ── shirt ──────────────────────────────────────────────────────────────
    "shirt":        "shirt",
    "blouse":       "shirt",
    "tee":          "shirt",
    "tees":         "shirt",
    "t-shirt":      "shirt",
    "camisole":     "shirt",
    "bustier":      "shirt",
    "corset":       "shirt",
    "halterneck":   "shirt",
    "bodysuit":     "shirt",
    "bra":          "shirt",   # sports bra / bra → shirt (closest category)
    "tankini":      "shirt",
    "polo":         "shirt",

    # ── sweater ────────────────────────────────────────────────────────────
    "sweater":      "sweater",
    "cardigan":     "sweater",
    "hoodie":       "sweater",
    "hoody":        "sweater",
    "sweatshirt":   "sweater",
    "pullover":     "sweater",
    "knitwear":     "sweater",
    "jumper":       "sweater",
    "crewneck":     "sweater",
    "turtleneck":   "sweater",
    "fleece":       "sweater",  # fleece top/jacket ambiguous? No — fleece alone = sweater
    "sherpa":       "sweater",

    # ── jacket ─────────────────────────────────────────────────────────────
    "jacket":       "jacket",
    "blazer":       "jacket",
    "gilet":        "jacket",
    "waistcoat":    "jacket",
    "bomber":       "jacket",
    "windbreaker":  "jacket",
    "anorak":       "jacket",
    "cagoule":      "jacket",
    "puffer":       "jacket",
    "overshirt":    "jacket",
    "shacket":      "jacket",
    "varsity":      "jacket",
    "letterman":    "jacket",

    # ── coat ───────────────────────────────────────────────────────────────
    "coat":         "coat",
    "overcoat":     "coat",
    "trench":       "coat",
    "peacoat":      "coat",
    "parka":        "coat",
    "raincoat":     "coat",
    "mackintosh":   "coat",

    # ── dress ──────────────────────────────────────────────────────────────
    "dress":        "dress",
    "gown":         "dress",
    "sundress":     "dress",
    "bodycon":      "dress",
    "kaftan":       "dress",
    "abaya":        "dress",
    "jalabiya":     "dress",
    "robe":         "dress",   # French: robe = dress

    # ── jumpsuit ───────────────────────────────────────────────────────────
    "jumpsuit":     "jumpsuit",
    "romper":       "jumpsuit",
    "playsuit":     "jumpsuit",
    "overalls":     "jumpsuit",
    "dungarees":    "jumpsuit",
    "catsuit":      "jumpsuit",
    "boilersuit":   "jumpsuit",
    "combinaison":  "jumpsuit",  # French

    # ── skirt ──────────────────────────────────────────────────────────────
    "skirt":        "skirt",
    "skorts":       "skirt",
    "skort":        "skirt",
    "jupe":         "skirt",    # French

    # ── pants ──────────────────────────────────────────────────────────────
    "pants":        "pants",
    "trousers":     "pants",
    "jeans":        "pants",
    "leggings":     "pants",
    "tights":       "pants",
    "joggers":      "pants",
    "sweatpants":   "pants",
    "trackpants":   "pants",
    "chinos":       "pants",
    "khakis":       "pants",
    "culottes":     "pants",
    "palazzos":     "pants",
    "pantalon":     "pants",    # French
    "pant":         "pants",

    # ── shorts ─────────────────────────────────────────────────────────────
    "shorts":       "shorts",
    "short":        "shorts",
    "trunks":       "shorts",
    "bermuda":      "shorts",   # French/international

    # ── shoes ──────────────────────────────────────────────────────────────
    "shoes":        "shoes",
    "shoe":         "shoes",
    "sneakers":     "shoes",
    "sneaker":      "shoes",
    "trainers":     "shoes",
    "boots":        "shoes",
    "boot":         "shoes",
    "sandals":      "shoes",
    "sandal":       "shoes",
    "slippers":     "shoes",
    "slipper":      "shoes",
    "loafers":      "shoes",
    "loafer":       "shoes",
    "heels":        "shoes",
    "pumps":        "shoes",
    "mules":        "shoes",
    "espadrilles":  "shoes",
    "wedges":       "shoes",
    "ballerinas":   "shoes",
    "clogs":        "shoes",
    "clog":         "shoes",
    "chaussures":   "shoes",    # French
    "bottes":       "shoes",    # French
    "footwear":     "shoes",

    # ── bag ────────────────────────────────────────────────────────────────
    "bag":          "bag",
    "bags":         "bag",
    "backpack":     "bag",
    "backpacks":    "bag",
    "handbag":      "bag",
    "handbags":     "bag",
    "tote":         "bag",
    "clutch":       "bag",
    "pouch":        "bag",
    "purse":        "bag",
    "satchel":      "bag",
    "duffel":       "bag",
    "duffle":       "bag",
    "holdall":      "bag",
    "sac":          "bag",     # French

    # ── glasses ────────────────────────────────────────────────────────────
    "glasses":      "glasses",
    "sunglasses":   "glasses",
    "eyewear":      "glasses",
    "goggles":      "glasses",
    "lunettes":     "glasses",  # French

    # ── hat ────────────────────────────────────────────────────────────────
    "hat":          "hat",
    "beanie":       "hat",
    "beret":        "hat",
    "snapback":     "hat",
    "visor":        "hat",
    "fedora":       "hat",
    "panama":       "hat",
    "bonnet":       "hat",     # French
    "casquette":    "hat",     # French
    "chapeau":      "hat",     # French

    # ── watch ──────────────────────────────────────────────────────────────
    "watch":        "watch",
    "watches":      "watch",
    "smartwatch":   "watch",
    "timepiece":    "watch",
    "chronograph":  "watch",
    "montre":       "watch",   # French

    # ── scarf ──────────────────────────────────────────────────────────────
    "scarf":        "scarf",
    "scarves":      "scarf",
    "shawl":        "scarf",
    "snood":        "scarf",
    "balaclava":    "scarf",
    "neckerchief":  "scarf",
    "pashmina":     "scarf",
    "écharpe":      "scarf",   # French
    "foulard":      "scarf",   # French

    # ── not_fashion ────────────────────────────────────────────────────────
    # Only include words that are NEVER fashion items
    "jibbitz":      "not_fashion",
    "charm":        "not_fashion",
    "shinguard":    "not_fashion",
    "shinguards":   "not_fashion",
    "gloves":       "not_fashion",   # goalkeeper/boxing gloves only
    "goalkeeper":   "not_fashion",
    "shin":         "not_fashion",
    "ball":         "not_fashion",
    "balls":        "not_fashion",
    "racket":       "not_fashion",
    "racquet":      "not_fashion",
    "helmet":       "not_fashion",
    "towel":        "not_fashion",
    "mat":          "not_fashion",
    "dumbbells":    "not_fashion",
    "weights":      "not_fashion",
    "flask":        "not_fashion",
}


# =============================================================================
# LABEL_DESCRIPTIONS
# Rich text used by sentence-transformers for semantic similarity.
# Handles ambiguous cases: "hiking pants", "ski jacket", "wool scarf" etc.
# No brand names — ST handles them via semantic proximity.
# =============================================================================

LABEL_DESCRIPTIONS = {

    "shirt": (
        "shirt blouse top t-shirt tee tank camisole tube bodysuit bra sports bra "
        "crop top halter halterneck bustier corset strapless sleeveless polo "
        "button down button up henley long sleeve short sleeve "
        "base layer undershirt thermal top rashguard swim top bikini top "
        "sports top gym top workout top activewear performance top "
        "chemise blouse débardeur haut brassière "
    ),

    "sweater": (
        "sweater cardigan hoodie sweatshirt pullover knitwear knit jumper "
        "crewneck turtleneck mock neck zip up quarter zip full zip fleece sherpa "
        "long cardigan cable knit chunky knit cashmere wool merino "
        "round neck pullover knit vest "
        "pull tricot sweat chandail "
    ),

    "jacket": (
        "jacket blazer gilet waistcoat vest puffer quilted padded "
        "windbreaker anorak rain jacket softshell hardshell "
        "bomber varsity letterman harrington trucker denim jacket leather jacket "
        "biker jacket moto jacket track jacket zip jacket fleece jacket "
        "suit jacket sport coat tuxedo jacket "
        "overshirt shacket field jacket utility jacket cargo jacket "
        "military jacket ski jacket snowboard jacket winter jacket "
        "body warmer sleeveless jacket "
        "veste blouson doudoune "
    ),

    "coat": (
        "coat overcoat trench peacoat duster longline coat "
        "wool coat cashmere coat wrap coat belted coat "
        "parka hooded coat duffle coat toggle coat raincoat mac "
        "faux fur coat shearling coat teddy coat cape poncho "
        "manteau imperméable pardessus "
    ),

    "dress": (
        "dress mini dress midi dress maxi dress gown evening gown "
        "sundress shirt dress wrap dress bodycon slip dress "
        "skater dress a-line dress smock dress tiered dress "
        "strapless dress halter dress one shoulder dress "
        "party dress occasion dress kaftan abaya "
        "robe robe courte robe longue "
    ),

    "jumpsuit": (
        "jumpsuit romper playsuit overalls dungarees one-piece "
        "boilersuit catsuit utility jumpsuit "
        "combinaison salopette "
    ),

    "skirt": (
        "skirt mini skirt midi skirt maxi skirt "
        "pleated skirt wrap skirt pencil skirt a-line skirt "
        "tiered skirt ruffle skirt flared skirt circle skirt "
        "denim skirt tennis skirt skort "
        "jupe jupe courte jupe longue "
    ),

    "pants": (
        "pants trousers jeans leggings tights pant "
        "wide leg straight leg slim barrel leg flare bootcut "
        "cropped pants joggers sweatpants trackpants "
        "cargo pants chinos chino khakis "
        "dress pants suit pants tailored trousers "
        "palazzo culottes parachute pants "
        "compression tights running tights cycling tights thermal tights "
        "hiking pants climbing pants ski pants snow pants "
        "pantalon jean legging collant "
    ),

    "shorts": (
        "shorts short cycling shorts swim shorts board shorts trunks "
        "running shorts gym shorts sport shorts "
        "denim shorts cargo shorts chino shorts biker shorts "
        "hiking shorts climbing shorts "
        "bermuda short "
    ),

    "shoes": (
        "shoes sneakers boots sandals slippers loafers heels pumps "
        "mules flip flops espadrilles wedges platforms ballerinas "
        "ankle boots chelsea boots combat boots knee high boots "
        "running shoes basketball shoes football boots tennis shoes "
        "hiking boots trail shoes hiking shoes "
        "clogs slip-on aqua shoes water shoes "
        "ski boots snowboard boots "
        "chaussures bottes sandales pantoufles footwear "
    ),

    "bag": (
        "bag backpack handbag tote shoulder bag crossbody "
        "clutch pouch purse satchel messenger bum bag "
        "fanny pack waist bag gym bag duffel duffle "
        "weekender travel bag holdall laptop bag "
        "mini bag belt bag "
        "sac sac à dos sacoche pochette "
    ),

    "glasses": (
        "glasses sunglasses eyewear goggles "
        "aviator wayfarer round cat eye oversized "
        "sports glasses swimming goggles ski goggles "
        "lunettes lunettes de soleil "
    ),

    "hat": (
        "hat cap beanie bucket hat beret snapback "
        "baseball cap trucker cap fitted cap visor "
        "sun hat straw hat fedora panama "
        "bobble hat ski hat winter hat "
        "headband hair accessory hair band "
        "chapeau bonnet casquette "
    ),

    "watch": (
        "watch smartwatch timepiece chronograph "
        "digital watch sports watch fitness tracker "
        "montre "
    ),

    "scarf": (
        "scarf scarves shawl snood balaclava neckerchief "
        "pashmina silk scarf wool scarf neck warmer "
        "écharpe foulard châle "
    ),

    "not_fashion": (
        "jibbitz charm decoration pin ornament figurine novelty clip "
        "ball football basketball tennis ball volleyball rugby ball "
        "shinguard shin guard shin pad knee pad elbow pad "
        "glove goalkeeper glove boxing glove batting glove "
        "racket racquet bat hockey stick "
        "helmet protection guard "
        "towel mat yoga mat water bottle flask "
        "resistance band jump rope skipping rope weights dumbbells "
        "sock socks pair pack "
        "underwear boxer brief trunk "
    ),
}


# =============================================================================
# YOLO_TO_CANONICAL
# =============================================================================

YOLO_TO_CANONICAL = {
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


# =============================================================================
# FASHIONPEDIA_TO_CANONICAL
# =============================================================================

FASHIONPEDIA_TO_CANONICAL = {
    "shirt, blouse":                           "shirt",
    "top, t-shirt, sweatshirt":                "shirt",
    "sweater":                                 "sweater",
    "cardigan":                                "sweater",
    "jacket":                                  "jacket",
    "vest":                                    "jacket",
    "coat":                                    "coat",
    "cape":                                    "coat",
    "pants":                                   "pants",
    "shorts":                                  "shorts",
    "skirt":                                   "skirt",
    "dress":                                   "dress",
    "jumpsuit":                                "jumpsuit",
    "glasses":                                 "glasses",
    "hat":                                     "hat",
    "headband, head covering, hair accessory": "hat",
    "watch":                                   "watch",
    "shoe":                                    "shoes",
    "bag, wallet":                             "bag",
    "scarf":                                   "scarf",
}


# =============================================================================
# SEARCHABLE_IDS
# =============================================================================

SEARCHABLE_IDS = {
    0,   # shirt, blouse
    1,   # top, t-shirt, sweatshirt
    2,   # sweater
    3,   # cardigan
    4,   # jacket
    5,   # vest
    6,   # pants
    7,   # shorts
    8,   # skirt
    9,   # coat
    10,  # dress
    11,  # jumpsuit
    12,  # cape
    13,  # glasses
    14,  # hat
    15,  # headband
    18,  # watch
    23,  # shoe
    24,  # bag, wallet
    25,  # scarf
}