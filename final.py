import streamlit as st
from PIL import Image
import plotly.graph_objects as go
from googletrans import Translator
import os
import tensorflow as tf
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Define Swish activation function
def swish(x):
    return x * tf.keras.backend.sigmoid(x)

# Register Swish as a custom object
tf.keras.utils.get_custom_objects().update({'swish': tf.keras.layers.Activation(swish)})

# Define the FixedDropout custom layer correctly
class FixedDropout(tf.keras.layers.Dropout):
    def __init__(self, rate, **kwargs):
        super(FixedDropout, self).__init__(rate, **kwargs)

# Register custom objects for loading model
custom_objects = {
    'swish': tf.keras.layers.Activation(swish),
    'FixedDropout': FixedDropout
}

@st.cache_resource
def load_model():
    model_path = "model/efficientmodel.h5"
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

model = load_model()


# ---------- Page Config ----------
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="centered")


# ---------- Custom Styles ----------
def set_styles():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

        .stApp {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(to right, #dcedc8, #f0f4c3, #c8e6c9);
            background-attachment: fixed;
            background-size: cover;
            color: #2e7d32;
        }

        .navbar {
            background: linear-gradient(to right, #388e3c, #43a047);
            background-color: #121212 !important;
            padding: 1rem;
            text-align: center;
            color: #ffffff !important;
            font-size: 28px;
            font-weight: bold;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            margin-bottom: 1.5rem;
        }
                
        [data-testid="stSidebar"] {
            background-color: #222222 !important;  /* dark gray */
            color: #ffffff !important;  /* white text */
        }

        /* Sidebar headings & text */
        [data-testid="stSidebar"] .css-1v3fvcr h2, 
        [data-testid="stSidebar"] .css-1v3fvcr p, 
        [data-testid="stSidebar"] .css-1v3fvcr div,
        [data-testid="stSidebar"] .css-1v3fvcr label {
            color: #ffffff !important;
        }

        h1, h3 {
            color: #1b5e20;
            text-align: center;
            font-weight: 600;
        }

        .stButton>button {
            background-color: #43a047;
            color: white;
            border-radius: 10px;
            padding: 0.6rem 2rem;
            font-weight: 600;
            border: none;
            transition: 0.3s ease;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        }

        .stButton>button:hover {
            background-color: #2e7d32;
            color: white;
        }

        .result-box {
            background-color: rgba(255, 255, 255, 0.95);
            padding: 1.2em;
            border-radius: 14px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
            margin-top: 25px;
            animation: fadeInUp 1s ease-in-out;
        }

        .uploaded-img {
            border: 3px solid #a5d6a7;
            border-radius: 10px;
            margin-top: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }

        .animated-heading {
            animation: fadeInUp 1.5s ease-in-out;
        }

        .result-box p {
            font-size: 18px;
            color: #444;
            line-height: 1.5;
        }

        hr {
            border: none;
            border-top: 2px dashed #a5d6a7;
            margin: 1.5rem 0;
        }

        @keyframes fadeInUp {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        </style>
    """, unsafe_allow_html=True)

set_styles()


# ---------- Navbar Title ----------
st.markdown("""
<div class='navbar animated-heading'>🌿 Plant Disease Detector</div>
""", unsafe_allow_html=True)


st.markdown("<p class='animated-heading' style='text-align: center; font-size: 18px;'>Upload a leaf image to detect diseases and get organic remedies</p>", unsafe_allow_html=True)


# ---------- Sidebar ----------
st.sidebar.title("📘 How to Use")
st.sidebar.markdown("""
1. Upload a clear crop leaf image  
2. Click 'Predict Disease'  
3. View disease & remedy  
4. Check model's confidence  
5. See remedy in your language  

> Ensure image is clear and in good lighting!
""")


# Button to open chatbot in a new tab
if st.sidebar.button("💬 Open Chatbot"):
    js = "window.open('http://localhost:8502')"  # Open chatbot at port 8502; adjust URL/port as needed
    st.components.v1.html(f"<script>{js}</script>")


# ---------- Language Dropdown ----------
language_map = {
    "English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te", "Marathi": "mr",
    "Bengali": "bn", "Gujarati": "gu", "Kannada": "kn", "Malayalam": "ml", "Urdu": "ur"
}
selected_language = st.selectbox("🌍 Select Your Language for Organic Remedy", list(language_map.keys()))


# ---------- Upload Image ----------
uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    center = st.columns([1, 2, 1])[1]
    with center:
        st.image(image, caption="🌱 Uploaded Leaf Image", width=250)
        predict_clicked = st.button("🔍 Predict Disease")

    if predict_clicked:
        with st.spinner("🔍 Analyzing the leaf..."):
            img = image.resize((224, 224))
            img_array = np.array(img).astype("float32")
            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions[0])

        # Ensure class names consistent with your model
        plant_village_classes = [
            'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
            'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
            'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
        ]

        remedies = {
            "Pepper__bell___Bacterial_spot": "🌶 Use copper-based organic sprays and avoid overhead watering.",
            "Pepper__bell___healthy": "✅ The leaf appears healthy!",
            "Potato___Early_blight": "🥔 Use crop rotation and apply bio-fungicides like neem oil.",
            "Potato___Late_blight": "🌧 Remove infected leaves, ensure good drainage, and apply organic copper sprays.",
            "Potato___healthy": "✅ The leaf appears healthy!",
            "Tomato___Target_Spot": "🎯 Remove affected leaves and apply bio-fungicides such as Trichoderma or neem oil.",
            "Tomato___Tomato_mosaic_virus": "🧬 Remove infected plants and sanitize tools. Use virus-free seeds.",
            "Tomato___Tomato_YellowLeaf__Curl_Virus": "🦠 Control whiteflies and remove infected plants promptly.",
            "Tomato_Bacterial_spot": "🦠 Apply copper-based sprays and avoid working with wet plants.",
            "Tomato_Early_blight": "🍅 Apply neem oil or copper fungicides weekly. Rotate crops regularly.",
            "Tomato_healthy": "✅ The leaf appears healthy!",
            "Tomato_Late_blight": "🌧 Prune infected parts and apply organic fungicides like potassium bicarbonate.",
            "Tomato_Leaf_Mold": "🍅 Improve air circulation and apply sulfur-based organic fungicides.",
            "Tomato_Septoria_leaf_spot": "🔬 Remove affected leaves, avoid overhead watering, and apply neem oil.",
            "Tomato_Spider_mites_Two_spotted_spider_mite": "🕷 Spray insecticidal soap or neem oil. Increase humidity around plants.",
            "default": "🌿 Use general organic care: neem oil, crop rotation, remove infected leaves."
        }

        result = plant_village_classes[predicted_index]
        remedy_text = remedies.get(result, remedies["default"])
        translator = Translator()
        translated_remedy = translator.translate(remedy_text, dest=language_map[selected_language]).text

        st.markdown(f"""
            <div class='result-box'>
                <h3>🦠 Disease Detected</h3>
                <p style="font-size:24px; font-weight:700; color:#d84315; text-align:center; margin-bottom:0.5rem;">{result.replace('_', ' ')}</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class='result-box'>
                <h3>🌿 Organic Remedy (English)</h3>
                <p style="text-align:center;">{remedy_text}</p>
            </div>
        """, unsafe_allow_html=True)

        if selected_language != "English":
            st.markdown(f"""
                <div class='result-box'>
                    <h3>🌐 Remedy in {selected_language}</h3>
                    <p style="text-align:center;">{translated_remedy}</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📊 Model Confidence per Disease")
        top_indices = predictions[0].argsort()[-5:][::-1]
        top_labels = [plant_village_classes[i] for i in top_indices]
        top_scores = [round(predictions[0][i] * 100, 2) for i in top_indices]

        fig = go.Figure(data=[
            go.Bar(
                x=top_labels,
                y=top_scores,
                marker_color=['#66bb6a', '#ffa726', '#ef5350', '#42a5f5', '#ab47bc'],
                text=[f"{s}%" for s in top_scores],
                textposition='auto'
            )
        ])
        fig.update_layout(
            title="Top 5 Predictions",
            title_x=0.5,
            yaxis_title='Confidence (%)',
            xaxis_title='Disease Class',
            plot_bgcolor='#f1f8e9',
            paper_bgcolor='#f1f8e9',
            font=dict(size=14),
            margin=dict(t=50, b=50),
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)

else:
    st.info("📷 Please upload a leaf image to get started.")
