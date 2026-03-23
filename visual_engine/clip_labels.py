# =============================================================================
# clip_labels.py — Single source of truth for all category labels
#
# 13 canonical categories (finalized):
#   top, sports_bra, pants, leggings, shorts, skirt, dress,
#   sweater, jacket, shoes, hat, bag, jumpsuit
#
# MATCHING APPROACH: exact token matching (split on spaces).
# Substring matching was considered but rejected — "hat" inside "that",
# "cap" inside "capsule", "bag" inside "baggy" all produce silent wrong
# classifications. When a word and its compound form both need to match
# (sleeve / sleeveless), add both explicitly.
# =============================================================================

CANONICAL_LABELS = [
    "top",
    "sports_bra",
    "pants",
    "leggings",
    "shorts",
    "skirt",
    "dress",
    "sweater",
    "jacket",
    "shoes",
    "hat",
    "bag",
    "jumpsuit",
]

# =============================================================================
# UNAMBIGUOUS_TOKEN_MAP
# =============================================================================

UNAMBIGUOUS_TOKEN_MAP = {

    # ── top ────────────────────────────────────────────────────────────────
    "shirt":        "top",
    "blouse":       "top",
    "tee":          "top",
    "tees":         "top",
    "tshirt":       "top",
    "t-shirt":      "top",
    "top":          "top",
    "tops":         "top",
    "camisole":     "top",
    "bustier":      "top",
    "corset":       "top",
    "halterneck":   "top",
    "bodysuit":     "top",
    "tankini":      "top",
    "polo":         "top",
    "tank":         "top",
    "sleeve":       "top",      # catches "long sleeve", "short sleeve"
    "sleeved":      "top",      # catches "short sleeved", "long sleeved"
    "sleeveless":   "top",
    "baselayer":    "top",
    "base":         "top",      # catches "base layer" after split
    "chemise":      "top",
    "débardeur":    "top",

    # ── sports_bra ─────────────────────────────────────────────────────────
    "bra":          "sports_bra",
    "bralette":     "sports_bra",
    "sports bra":   "sports_bra",
    "sport bra":    "sports_bra",
    "brassière":    "sports_bra",

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
    "fleece":       "sweater",
    "sherpa":       "sweater",
    "pull":         "sweater",
    "tricot":       "sweater",

    # ── jacket ─────────────────────────────────────────────────────────────
    "jacket":       "jacket",
    "blazer":       "jacket",
    "gilet":        "jacket",
    "waistcoat":    "jacket",
    "vest":         "jacket",
    "bomber":       "jacket",
    "windbreaker":  "jacket",
    "anorak":       "jacket",
    "cagoule":      "jacket",
    "puffer":       "jacket",
    "overshirt":    "jacket",
    "shacket":      "jacket",
    "varsity":      "jacket",
    "letterman":    "jacket",
    "coat":         "jacket",
    "overcoat":     "jacket",
    "trench":       "jacket",
    "peacoat":      "jacket",
    "parka":        "jacket",
    "raincoat":     "jacket",
    "mackintosh":   "jacket",
    "veste":        "jacket",
    "blouson":      "jacket",
    "doudoune":     "jacket",
    "manteau":      "jacket",
    "imperméable":  "jacket",

    # ── dress ──────────────────────────────────────────────────────────────
    "dress":        "dress",
    "gown":         "dress",
    "sundress":     "dress",
    "bodycon":      "dress",
    "kaftan":       "dress",
    "abaya":        "dress",
    "jalabiya":     "dress",
    "robe":         "dress",

    # ── jumpsuit ───────────────────────────────────────────────────────────
    "jumpsuit":     "jumpsuit",
    "romper":       "jumpsuit",
    "playsuit":     "jumpsuit",
    "overalls":     "jumpsuit",
    "dungarees":    "jumpsuit",
    "catsuit":      "jumpsuit",
    "boilersuit":   "jumpsuit",
    "combinaison":  "jumpsuit",
    "salopette":    "jumpsuit",

    # ── skirt ──────────────────────────────────────────────────────────────
    "skirt":        "skirt",
    "skorts":       "skirt",
    "skort":        "skirt",
    "jupe":         "skirt",

    # ── pants ──────────────────────────────────────────────────────────────
    "pants":        "pants",
    "trousers":     "pants",
    "jeans":        "pants",
    "joggers":      "pants",
    "sweatpants":   "pants",
    "trackpants":   "pants",
    "chinos":       "pants",
    "chino":        "pants",
    "khakis":       "pants",
    "culottes":     "pants",
    "palazzos":     "pants",
    "pantalon":     "pants",
    "pant":         "pants",

    # ── leggings ───────────────────────────────────────────────────────────
    "leggings":     "leggings",
    "legging":      "leggings",
    "tight":        "leggings",
    "tights":       "leggings",
    "collant":      "leggings",

    # ── shorts ─────────────────────────────────────────────────────────────
    "shorts":       "shorts",
    "short":        "shorts",
    "trunks":       "shorts",
    "bermuda":      "shorts",

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
    "chaussures":   "shoes",
    "bottes":       "shoes",
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
    "sacoche":      "bag",
    "pochette":     "bag",

    # ── hat ────────────────────────────────────────────────────────────────
    "hat":          "hat",
    "cap":          "hat",
    "beanie":       "hat",
    "beret":        "hat",
    "snapback":     "hat",
    "visor":        "hat",
    "fedora":       "hat",
    "panama":       "hat",
    "casquette":    "hat",
    "chapeau":      "hat",

    # ── not_fashion ────────────────────────────────────────────────────────
    "jibbitz":      "not_fashion",
    "shinguard":    "not_fashion",
    "shinguards":   "not_fashion",
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
    "swimsuit":     "not_fashion",
    "bikini":       "not_fashion",
    "swimwear":     "not_fashion",
    "swimshirt":    "not_fashion",
    "socks":        "not_fashion",
    "sock":         "not_fashion",
    "underwear":    "not_fashion",
    "boxer":        "not_fashion",
    "briefs":       "not_fashion",
}


# =============================================================================
# LABEL_DESCRIPTIONS
#
# Used by sentence-transformers for semantic similarity (fallback only —
# whitelist hits never reach here). Each description must be tight and
# unambiguous. Rules applied in this version:
#
#   - No words from other categories' token maps (e.g. no "shorts" in leggings)
#   - No swimwear-adjacent words (rashguard removed from top)
#   - No "knit vest" in sweater — vest now maps to jacket
#   - No headband/hair accessory in hat — not hats
#   - No "swim shorts" in shorts — swimwear adjacent
#   - No "trunk" in not_fashion — trunks maps to shorts
#   - Kept only words that unambiguously describe the category
# =============================================================================

LABEL_DESCRIPTIONS = {

    "top": (
        "top t-shirt tee tank camisole tube crop top "
        "halter halterneck polo button down button up henley "
        "long sleeve short sleeve sleeveless strapless "
        "base layer baselayer thermal top performance top gym top workout top "
        "fitted top casual top everyday top "
        "débardeur haut "
        # NOTE: "shirt" and "blouse" intentionally omitted — they are in the
        # token map and whitelist hits never reach this description.
        # Adding them here risks pulling shirt/blouse products toward top
        # when the token map already handles them definitively.
    ),

    "sports_bra": (
        "sports bra bralette athletic bra workout bra gym bra "
        "high impact sports bra medium support bra low impact bra "
        "crop sports bra padded sports bra zip front sports bra "
        "activewear bra fitness bra running bra yoga bra "
        "brassière de sport "
    ),

    "sweater": (
        "sweater cardigan hoodie sweatshirt pullover knitwear knit jumper "
        "crewneck turtleneck mock neck zip up quarter zip full zip "
        "fleece sherpa long cardigan cable knit chunky knit "
        "cashmere wool merino round neck "
        # Removed: "knit vest" — vest now maps to jacket, keeping it here
        # would pull jacket-voted products toward sweater incorrectly.
        "pull tricot sweat chandail "
    ),

    "jacket": (
        "jacket blazer gilet waistcoat vest puffer quilted padded "
        "windbreaker anorak rain jacket softshell hardshell "
        "bomber varsity letterman trucker denim jacket leather jacket "
        "biker jacket moto jacket track jacket zip jacket "
        "suit jacket sport coat tuxedo jacket "
        "overshirt shacket field jacket utility jacket cargo jacket "
        "military jacket ski jacket snowboard jacket "
        "coat overcoat trench peacoat duster longline coat "
        "wool coat cashmere coat wrap coat belted coat "
        "parka hooded coat duffle coat raincoat mac "
        "faux fur coat shearling coat teddy coat cape poncho "
        "veste blouson doudoune manteau imperméable "
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
        "jumpsuit romper playsuit overalls dungarees one-piece outfit "
        "boilersuit catsuit utility jumpsuit wide leg jumpsuit "
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
        "pants trousers jeans pant wide leg straight leg slim "
        "barrel leg flare bootcut cropped pants "
        "joggers sweatpants trackpants cargo pants "
        "chinos chino khakis dress pants suit pants tailored trousers "
        "palazzo culottes parachute pants "
        "pantalon jean "
    ),

    "leggings": (
        "leggings tights tight running tights compression tights "
        "thermal tights yoga pants gym leggings workout leggings "
        "high waist leggings seamless leggings printed leggings "
        "athletic leggings performance leggings activewear leggings "
        "collant legging "
        # Removed: "biker shorts", "tight shorts", "cycling shorts", "gym shorts"
        # These are shorts words — their presence was pulling shorts-voted
        # products toward leggings and vice versa.
    ),

    "shorts": (
        "shorts denim shorts cargo shorts chino shorts "
        "running shorts sport shorts board shorts "
        "hiking shorts casual shorts bermuda trunks "
        # Removed: "swim shorts" — swimwear adjacent, better to skip than misclassify
    ),

    "shoes": (
        "shoes sneakers boots sandals slippers loafers heels pumps "
        "mules flip flops espadrilles wedges platforms ballerinas "
        "ankle boots chelsea boots combat boots knee high boots "
        "running shoes basketball shoes tennis shoes "
        "hiking boots trail shoes clogs slip-on "
        "ski boots snowboard boots "
        "chaussures bottes sandales footwear "
    ),

    "bag": (
        "bag backpack handbag tote shoulder bag crossbody "
        "clutch pouch purse satchel messenger bum bag "
        "fanny pack waist bag gym bag duffel duffle "
        "weekender travel bag holdall laptop bag "
        "mini bag belt bag "
        "sac sac à dos sacoche pochette "
    ),

    "hat": (
        "hat cap beanie bucket hat beret snapback "
        "baseball cap trucker cap fitted cap visor "
        "sun hat straw hat fedora panama "
        "bobble hat ski hat winter hat "
        "chapeau bonnet casquette "
        # Removed: "headband hair accessory hair band" — not hats.
        # These were pulling ambiguous titles toward hat incorrectly.
    ),

    "not_fashion": (
        "swimsuit bikini one piece swimwear bathing suit "
        "jibbitz decoration pin novelty "
        "ball football basketball tennis ball volleyball rugby "
        "shinguard shin guard shin pad knee pad elbow pad "
        "racket racquet bat hockey stick "
        "helmet protection guard "
        "towel mat yoga mat water bottle flask "
        "resistance band jump rope weights dumbbells "
        "socks underwear boxer brief "
        # Removed: "trunk" — trunks maps to shorts, not not_fashion
        "glove goalkeeper glove boxing glove "
    ),
}


# =============================================================================
# YOLO_TO_CANONICAL
# =============================================================================

YOLO_TO_CANONICAL = {
    "short sleeved shirt":   "top",
    "long sleeved shirt":    "top",
    "short sleeved outwear": "jacket",
    "long sleeved outwear":  "jacket",
    "vest":                  "jacket",
    "sling":                 "top",
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
    "shirt, blouse":                           "top",
    "top, t-shirt, sweatshirt":                "top",
    "sweater":                                 "sweater",
    "cardigan":                                "sweater",
    "jacket":                                  "jacket",
    "vest":                                    "jacket",
    "coat":                                    "jacket",
    "cape":                                    "jacket",
    "pants":                                   "pants",
    "shorts":                                  "shorts",
    "skirt":                                   "skirt",
    "dress":                                   "dress",
    "jumpsuit":                                "jumpsuit",
    "hat":                                     "hat",
    "headband, head covering, hair accessory": "hat",
    "shoe":                                    "shoes",
    "bag, wallet":                             "bag",
}


# =============================================================================
# SEARCHABLE_IDS
# =============================================================================

SEARCHABLE_IDS = {
    0,   # shirt, blouse       → top
    1,   # top, t-shirt        → top
    2,   # sweater             → sweater
    3,   # cardigan            → sweater
    4,   # jacket              → jacket
    5,   # vest                → jacket
    6,   # pants               → pants
    7,   # shorts              → shorts
    8,   # skirt               → skirt
    9,   # coat                → jacket
    10,  # dress               → dress
    11,  # jumpsuit            → jumpsuit
    12,  # cape                → jacket
    14,  # hat                 → hat
    15,  # headband            → hat
    23,  # shoe                → shoes
    24,  # bag, wallet         → bag
}