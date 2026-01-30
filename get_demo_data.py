import requests
import os

# Config
DEST_FOLDER = "demo_images"

# 50 Items (Expanded to fix Ranking issues)
# We added more "Red Dresses" and "casual wear" to crowd out the purses.
image_map = {
    # --- ZARA (Dresses - Now with color variety for ranking tests) ---
    "zara_red_dress.jpg": "https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=500",
    "zara_red_maxi_dress.jpg": "https://images.unsplash.com/photo-1518831959646-742c3a14ebf7?w=500", # New
    "zara_red_slip_dress.jpg": "https://images.unsplash.com/photo-1566174053879-31528523f8ae?w=500", # New
    "zara_black_dress.jpg": "https://images.unsplash.com/photo-1539008835657-9e8e9680c956?w=500",
    "zara_little_black_dress.jpg": "https://images.unsplash.com/photo-1544441893-675973e31985?w=500", # New
    "zara_white_blouse.jpg": "https://images.unsplash.com/photo-1564257631407-4deb1f99d992?w=500",
    "zara_white_summer_dress.jpg": "https://images.unsplash.com/photo-1515372039744-b8f02a3ae446?w=500", # New
    "zara_summer_floral.jpg": "https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?w=500",
    "zara_floral_mini.jpg": "https://images.unsplash.com/photo-1551163943-3f6a29e39c66?w=500", # New
    "zara_evening_gown.jpg": "https://images.unsplash.com/photo-1566174053879-31528523f8ae?w=500",
    "zara_polka_dot_dress.jpg": "https://images.unsplash.com/photo-1502716119720-b23a93e5fe1b?w=500", # New

    # --- BERSHKA (Casual, Denim & Streetwear) ---
    "bershka_blue_jeans.jpg": "https://images.unsplash.com/photo-1541099649105-f69ad21f3246?w=500",
    "bershka_mom_jeans.jpg": "https://images.unsplash.com/photo-1542272454315-4c01d7abdf4a?w=500", # New
    "bershka_denim_jacket.jpg": "https://images.unsplash.com/photo-1543163521-1bf539c55dd2?w=500",
    "bershka_ripped_shorts.jpg": "https://images.unsplash.com/photo-1582552938357-32b906df40cb?w=500",
    "bershka_graphic_tee.jpg": "https://images.unsplash.com/photo-1503341504253-dff4815485f1?w=500",
    "bershka_oversized_hoodie.jpg": "https://images.unsplash.com/photo-1556905055-8f358a7a47b2?w=500", # New
    "bershka_streetwear.jpg": "https://images.unsplash.com/photo-1552374196-1ab2a1c593e8?w=500",
    "bershka_cargo_pants.jpg": "https://images.unsplash.com/photo-1565538420115-4e782622956c?w=500", # New
    "bershka_crop_top.jpg": "https://images.unsplash.com/photo-1509631179647-0177331693ae?w=500", # New
    "bershka_leather_skirt.jpg": "https://images.unsplash.com/photo-1534030634647-8a4362b083c7?w=500", # New

    # --- MIKE SPORT (Athletic) ---
    "mike_sport_red_sneakers.jpg": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=500",
    "mike_sport_running_shoes.jpg": "https://images.unsplash.com/photo-1521774971864-62e842046145?w=500",
    "mike_sport_gym_leggings.jpg": "https://images.unsplash.com/photo-1534438327276-14e5300c3a48?w=500",
    "mike_sport_yoga_pants.jpg": "https://images.unsplash.com/photo-1506619216599-9d16d0903dfd?w=500", # New
    "mike_sport_white_kicks.jpg": "https://images.unsplash.com/photo-1560769629-975e13b01e5e?w=500",
    "mike_sport_training_top.jpg": "https://images.unsplash.com/photo-1518310383802-640c2de311b2?w=500",
    "mike_sport_sports_bra.jpg": "https://images.unsplash.com/photo-1571945153262-cb4cfce391d1?w=500", # New
    "mike_sport_track_jacket.jpg": "https://images.unsplash.com/photo-1595341888016-a392ef81b7de?w=500", # New
    "mike_sport_basketball_shorts.jpg": "https://images.unsplash.com/photo-1519077203671-554477b5a1b3?w=500", # New

    # --- LOUIS VUITTON (Bags & Luxury) ---
    "louis_vuitton_leather_tote.jpg": "https://images.unsplash.com/photo-1584917865442-de89df76afd3?w=500",
    "louis_vuitton_clutch.jpg": "https://images.unsplash.com/photo-1566150905458-1bf1fc113f0d?w=500",
    "louis_vuitton_handbag.jpg": "https://images.unsplash.com/photo-1590874103328-eac38a683ce7?w=500",
    "louis_vuitton_red_purse.jpg": "https://images.unsplash.com/photo-1566150902878-574347209ca4?w=500", # New - To test color confusion
    "louis_vuitton_travel_bag.jpg": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=500",
    "louis_vuitton_brown_purse.jpg": "https://images.unsplash.com/photo-1548036328-c9fa89d128fa?w=500",
    "louis_vuitton_wallet.jpg": "https://images.unsplash.com/photo-1627123424574-724758594e93?w=500", # New
    "louis_vuitton_belt.jpg": "https://images.unsplash.com/photo-1624222247344-550fb60583dc?w=500", # New

    # --- VIRGIN (Electronics - To prove we can handle non-fashion) ---
    "virgin_headphones.jpg": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=500",
    "virgin_earbuds.jpg": "https://images.unsplash.com/photo-1590658268037-6bf12165a8df?w=500",
    "virgin_speaker.jpg": "https://images.unsplash.com/photo-1545454675-3531b543be5d?w=500",
    "virgin_smartwatch.jpg": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=500",
    "virgin_camera.jpg": "https://images.unsplash.com/photo-1516035069371-29a1b244cc32?w=500",
    "virgin_drone.jpg": "https://images.unsplash.com/photo-1506947411487-a56738267384?w=500", # New
    "virgin_gaming_mouse.jpg": "https://images.unsplash.com/photo-1527814050087-3793815479db?w=500" # New
}

def fetch_images():
    if not os.path.exists(DEST_FOLDER):
        os.makedirs(DEST_FOLDER)
        print(f"📂 Created folder: {DEST_FOLDER}")

    print(f"🚀 Downloading {len(image_map)} demo images...")
    
    success_count = 0
    for filename, url in image_map.items():
        path = os.path.join(DEST_FOLDER, filename)
        
        # Don't re-download if exists
        if os.path.exists(path):
            print(f"  Example exists: {filename}")
            success_count += 1
            continue
            
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(path, "wb") as f:
                    f.write(r.content)
                print(f"  ✅ Downloaded: {filename}")
                success_count += 1
            else:
                print(f"  ❌ Failed: {filename} (Status {r.status_code})")
        except Exception as e:
            print(f"  ❌ Error {filename}: {e}")

    print(f"\n✨ Data fetch complete! {success_count}/{len(image_map)} images ready.")
    print("👉 NOW RUN: & 'c:\\Python314\\python.exe' 'c:\\dev\\locus\\bulk_upload.py'")

if __name__ == "__main__":
    fetch_images()