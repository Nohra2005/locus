import streamlit as st
import requests
from PIL import Image
import os

st.set_page_config(layout="wide")
st.title("LOCUS: Mall Visual Search Debugger")

# 1. Image Upload
uploaded_file = st.file_uploader("Upload a photo to search...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the query image
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image(uploaded_file, caption="Query Image", use_container_width=True) # <--- FIXED
    
    # 2. Send to API
    with st.spinner("Analyzing visuals & Searching Inventory..."):
        files = {"file": uploaded_file.getvalue()}
        try:
            # Note: We use the Gateway URL directly
            response = requests.post("http://localhost:8000/search", files=files)
            # We look for "matches" because that is what your API returns now
            results = response.json().get("matches", [])
        except Exception as e:
            st.error(f"Connection Error: {e}")
            results = []

    # 3. Display Results in a Grid
    if not results:
        st.warning("Found 0 Matches. (Check if your Docker containers are running)")
    else:
        st.subheader(f"Found {len(results)} Matches")
        
        # Create rows of 4 images
        cols = st.columns(4)
        for idx, item in enumerate(results):
            # OPTIONAL: Hide results that are less than 75% similar
            # if item['score'] < 0.75: continue 

            with cols[idx % 4]:
                # Construct the path to the local image
                # We assume the images are in the 'demo_images' folder relative to this script
                local_image_path = os.path.join("demo_images", item['image_filename'])
                
                try:
                    # Check if file exists first to avoid crashing
                    if os.path.exists(local_image_path):
                        img = Image.open(local_image_path)
                        st.image(img, use_container_width=True) # <--- FIXED
                        
                        # Display Metadata clearly
                        st.markdown(f"**{item['store']}** ({item['level']})")
                        st.caption(f"Score: {item['score']:.2f} | {item['name']}")
                    else:
                        st.error(f"Image not found: {item['image_filename']}")
                except Exception as e:
                    st.warning(f"Could not load image: {e}")