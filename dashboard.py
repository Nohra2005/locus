import streamlit as st
import requests
from PIL import Image
import io
import base64
import os
from streamlit_cropper import st_cropper 

st.set_page_config(layout="wide", page_title="Locus Lens")

st.title("🔎 Locus Lens: Select & Search")
st.markdown("Upload a photo, **draw a box** around the item you want, and hit Search.")

# 1. Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    
    # [PARTNER LOGIC] The Cropping Tool
    st.subheader("1. Select the Item")
    cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
    st.caption("Adjust the red box to frame the item perfectly.")

    # Search Button
    if st.button("🔍 Search This Selection", type="primary"):
        
        # Convert crop to bytes
        img_byte_arr = io.BytesIO()
        cropped_img.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        st.divider()

        with st.spinner("🚀 AI is analyzing your selection..."):
            try:
                # Send the CROPPED image to the backend
                files = {"file": ("crop.png", img_bytes, "image/png")}
                response = requests.post("http://localhost:8000/search", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    matches = data.get("matches", [])
                    debug_image_b64 = data.get("debug_image")
                    
                    # [YOUR LOGIC] The Category Banner
                    category = data.get("detected_category")
                    if category:
                        st.success(f"👁️ AI Identified: **{category.upper()}** (Filtering active)")
                    else:
                        st.warning("👁️ AI Identified: **Unsure** (Showing all categories)")

                    # [PARTNER LOGIC] Show AI Vision
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Your Selection")
                        st.image(cropped_img, caption="Cropped Input", width=300)
                    
                    with col2:
                        st.subheader("AI Vision")
                        if debug_image_b64:
                            image_data = base64.b64decode(debug_image_b64)
                            ai_image = Image.open(io.BytesIO(image_data))
                            st.image(ai_image, caption="Background Removed", width=300)

                    # Results Grid
                    st.header(f"🎯 Top Matches ({len(matches)})")
                    if matches:
                        cols = st.columns(5)
                        for idx, item in enumerate(matches):
                            with cols[idx % 5]:
                                local_path = os.path.join("demo_images", item['image_filename'])
                                if os.path.exists(local_path):
                                    st.image(local_path, use_container_width=True)
                                    
                                    # Info
                                    if idx == 0: st.markdown(f"**🥇 {item['name']}**")
                                    else: st.markdown(f"**{item['name']}**")
                                    
                                    score = item['score']
                                    color = "green" if score > 0.8 else "orange"
                                    st.markdown(f":{color}[Score: {score:.3f}]")
                                    st.caption(f"{item['store']} • {item['level']}")
                                else:
                                    st.error(f"Missing: {item['image_filename']}")
                    else:
                        st.warning("No matches found.")
                else:
                    st.error(f"Backend Error: {response.status_code}")

            except Exception as e:
                st.error(f"Connection Error: {e}")