import streamlit as st
import numpy as np
from PIL import Image
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

# Define MAC complexion RGB values
mac_complexion_rgb = {
    "NC10": (255, 224, 198), "NC15": (255, 218, 185), "NC20": (255, 205, 170), "NC25": (255, 190, 155),
    "NC30": (230, 170, 140), "NC35": (210, 150, 120), "NC40": (190, 130, 100), "NC42": (170, 110, 80),
    "NC44": (150, 90, 60), "NC45": (130, 70, 50), "NC50": (110, 50, 30), "NC55": (90, 30, 10)
}

# Create KDTree for MAC complexions
complexion_tree = KDTree(list(mac_complexion_rgb.values()))

def extract_skin_tone(image):
    img_array = np.array(image)
    height, width, _ = img_array.shape
    center_h, center_w = height // 2, width // 2
    sample_h, sample_w = max(1, height // 10), max(1, width // 10)
    sample = img_array[center_h-sample_h:center_h+sample_h, center_w-sample_w:center_w+sample_w]
    avg_color = np.mean(sample, axis=(0, 1))
    return tuple(int(c) for c in avg_color)

def find_mac_complexion(rgb_value):
    _, idx = complexion_tree.query(rgb_value)
    return list(mac_complexion_rgb.keys())[idx]

lipstick_recommendations = {
    "NC10": ["Angel", "Creme Cup", "Snob"], "NC15": ["Brave", "Twig", "Mehr"], "NC20": ["Velvet Teddy", "Mehr", "Mocha"],
    "NC25": ["Taupe", "Whirl", "Mocha"], "NC30": ["Whirl", "Mocha", "Chili"], "NC35": ["Chili", "Marrakesh", "Brick-O-La"],
    "NC40": ["Chili", "Russian Red", "Marrakesh"], "NC42": ["Del Rio", "Sin", "Diva"], "NC44": ["Diva", "Sin", "Media"],
    "NC45": ["Media", "Ruby Woo", "Heroine"], "NC50": ["Media", "Heroine", "Rebel"], "NC55": ["Smoked Purple", "Diva", "Ruby Woo"]
}

st.title("MAC Lipstick Recommender 💄")
st.write("Upload your photo to get the best MAC lipstick shades for your skin tone!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    skin_tone_rgb = extract_skin_tone(image)
    mac_complexion = find_mac_complexion(skin_tone_rgb)
    recommendations = lipstick_recommendations.get(mac_complexion, ["No recommendations available"])
    
    st.subheader("Results:")
    st.write(f"**Detected RGB:** {skin_tone_rgb}")
    st.write(f"**MAC Complexion Match:** {mac_complexion}")
    
    st.write("### Recommended MAC Lipsticks:")
    for lipstick in recommendations:
        st.write(f"- {lipstick}")
    
    # Display skin tone swatches
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2))
    ax1.add_patch(plt.Rectangle((0, 0), 1, 1, color=[c/255 for c in skin_tone_rgb]))
    ax1.set_title("Detected Skin Tone")
    ax1.axis('off')
    
    mac_color = mac_complexion_rgb[mac_complexion]
    ax2.add_patch(plt.Rectangle((0, 0), 1, 1, color=[c/255 for c in mac_color]))
    ax2.set_title(f"MAC {mac_complexion}")
    ax2.axis('off')
    
    st.pyplot(fig)
    
    st.write("_Note: For best results, take photos in natural lighting with a neutral background._")
