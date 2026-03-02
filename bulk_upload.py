import requests
import os
import pandas as pd
import random

# ─── Config ───────────────────────────────────────────────────────────────────
API_URL      = "http://localhost:8000/add"
DATASET_PATH = r"C:\Users\User\Downloads\myntradataset"
IMAGES_PATH  = os.path.join(DATASET_PATH, "images")
CSV_PATH     = os.path.join(DATASET_PATH, "styles.csv")
MALL_NAME    = "ABC Achrafieh"
SAMPLE_SIZE  = 200   # how many items to upload (increase later if needed)

# ─── Store Mapping ────────────────────────────────────────────────────────────
# Maps Myntra articleType → which store in the mall carries it
# Based on your mall_config.json
STORE_MAP = {
    # ZARA — Dresses, Tops, Coats, Blazers
    "Dresses":          {"store": "Zara",          "level": "L2"},
    "Tops":             {"store": "Zara",          "level": "L2"},
    "Blouses":          {"store": "Zara",          "level": "L2"},
    "Shirts":           {"store": "Zara",          "level": "L2"},
    "Coats":            {"store": "Zara",          "level": "L2"},
    "Jackets":          {"store": "Zara",          "level": "L2"},
    "Blazers":          {"store": "Zara",          "level": "L2"},
    "Skirts":           {"store": "Zara",          "level": "L2"},
    "Sweaters":         {"store": "Zara",          "level": "L2"},

    # BERSHKA — Casual, Denim, Streetwear
    "Jeans":            {"store": "Bershka",       "level": "L1"},
    "Trousers":         {"store": "Bershka",       "level": "L1"},
    "Shorts":           {"store": "Bershka",       "level": "L1"},
    "Sweatshirts":      {"store": "Bershka",       "level": "L1"},
    "Hoodies":          {"store": "Bershka",       "level": "L1"},
    "Tshirts":          {"store": "Bershka",       "level": "L1"},
    "Casual Shirts":    {"store": "Bershka",       "level": "L1"},
    "Track Pants":      {"store": "Bershka",       "level": "L1"},
    "Leggings":         {"store": "Bershka",       "level": "L1"},

    # MIKE SPORT — Shoes, Activewear
    "Sports Shoes":     {"store": "Mike Sport",    "level": "L3"},
    "Casual Shoes":     {"store": "Mike Sport",    "level": "L3"},
    "Sneakers":         {"store": "Mike Sport",    "level": "L3"},
    "Running Shoes":    {"store": "Mike Sport",    "level": "L3"},
    "Sports Sandals":   {"store": "Mike Sport",    "level": "L3"},
    "Sports Bra":       {"store": "Mike Sport",    "level": "L3"},
    "Tracksuits":       {"store": "Mike Sport",    "level": "L3"},

    # LOUIS VUITTON — Bags, Accessories
    "Handbags":         {"store": "Louis Vuitton", "level": "L0"},
    "Clutches":         {"store": "Louis Vuitton", "level": "L0"},
    "Backpacks":        {"store": "Louis Vuitton", "level": "L0"},
    "Wallets":          {"store": "Louis Vuitton", "level": "L0"},
    "Belts":            {"store": "Louis Vuitton", "level": "L0"},
    "Sandals":          {"store": "Louis Vuitton", "level": "L0"},
    "Heels":            {"store": "Louis Vuitton", "level": "L0"},
    "Flats":            {"store": "Louis Vuitton", "level": "L0"},

    # VIRGIN — Tech Accessories
    "Sunglasses":       {"store": "Virgin",        "level": "L4"},
    "Watches":          {"store": "Virgin",        "level": "L4"},
    "Eyewear Frames":   {"store": "Virgin",        "level": "L4"},
}

# ─── Load CSV ─────────────────────────────────────────────────────────────────
def load_dataset():
    print(f"📂 Loading {CSV_PATH}...")

    # Myntra CSV has some bad rows — on_bad_lines skips them
    df = pd.read_csv(CSV_PATH, on_bad_lines="skip")

    print(f"   Total rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")

    # Keep only rows whose articleType is in our store map
    df = df[df["articleType"].isin(STORE_MAP.keys())]
    print(f"   After filtering to known categories: {len(df)} rows")

    # Keep only rows whose image file actually exists
    df["image_path"] = df["id"].apply(
        lambda x: os.path.join(IMAGES_PATH, f"{x}.jpg")
    )
    df = df[df["image_path"].apply(os.path.exists)]
    print(f"   After checking image files exist: {len(df)} rows")

    return df

# ─── Sample evenly across categories ─────────────────────────────────────────
def sample_balanced(df, total=SAMPLE_SIZE):
    """
    Take an even spread across article types so Qdrant has good coverage.
    e.g. if we have 10 categories and want 200 items → ~20 per category.
    """
    categories = df["articleType"].unique()
    per_cat    = max(1, total // len(categories))

    sampled = []
    for cat in categories:
        subset = df[df["articleType"] == cat]
        n      = min(per_cat, len(subset))
        sampled.append(subset.sample(n, random_state=42))

    result = pd.concat(sampled).sample(frac=1, random_state=42)  # shuffle
    print(f"\n📊 Sampling {len(result)} items across {len(categories)} categories:")
    for cat in categories:
        count = len(result[result["articleType"] == cat])
        print(f"   {cat:<25} → {count} items")

    return result

# ─── Upload ───────────────────────────────────────────────────────────────────
def run_upload(df):
    print(f"\n🚀 Starting upload of {len(df)} items to {MALL_NAME}...")

    success = 0
    failed  = 0
    skipped = 0

    for _, row in df.iterrows():
        item_id    = row["id"]
        name       = str(row.get("productDisplayName", f"Item {item_id}"))
        article    = row["articleType"]
        colour     = str(row.get("baseColour", ""))
        store_info = STORE_MAP[article]
        store      = store_info["store"]
        level      = store_info["level"]
        image_path = row["image_path"]
        filename   = os.path.basename(image_path)

        # Make a clean display name: "Blue Casual Shirt" instead of raw DB name
        display_name = name if name and name != "nan" else f"{colour} {article}".strip()

        print(f"  📤 [{store}] {display_name[:50]}...")

        try:
            with open(image_path, "rb") as img:
                r = requests.post(
                    API_URL,
                    data={
                        "name":  display_name,
                        "store": store,
                        "level": level,
                        "mall":  MALL_NAME,
                    },
                    files={"file": (filename, img, "image/jpeg")},
                    timeout=60
                )

            if r.status_code == 200:
                print(f"     ✅ Saved")
                success += 1
            elif r.status_code == 400:
                # This means the visual engine rejected the image
                # (ghost image, low confidence, etc.)
                print(f"     ⚠️  Skipped (visual engine rejected): {r.text}")
                skipped += 1
            else:
                print(f"     ❌ Failed ({r.status_code}): {r.text}")
                failed += 1

        except Exception as e:
            print(f"     ❌ Error: {e}")
            failed += 1

    print(f"\n🏁 Done!")
    print(f"   ✅ Uploaded: {success}")
    print(f"   ⚠️  Skipped:  {skipped}  (rejected by visual engine — normal)")
    print(f"   ❌ Failed:   {failed}")
    print(f"\n👉 Open http://localhost:8501 and test your search!")

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Step 1: check dataset folder exists
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at: {DATASET_PATH}")
        print("   Check the path and try again.")
        exit(1)

    # Step 2: load and filter CSV
    df = load_dataset()

    if len(df) == 0:
        print("❌ No valid items found. Check that images/ folder exists.")
        exit(1)

    # Step 3: balanced sample
    df_sample = sample_balanced(df, total=SAMPLE_SIZE)

    # Step 4: upload
    run_upload(df_sample)