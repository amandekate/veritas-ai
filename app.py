import streamlit as st
import tensorflow as tf
import tf_keras
import keras
import numpy as np
from PIL import Image
from transformers import BertTokenizer, TFBertModel
from tf_keras.applications import resnet50

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Veritas AI | Multimodal Fake News Detector",
    page_icon="🕵️‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (Added Weights Box Styling)
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        height: 3em;
        font-size: 20px;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover { background-color: #ff3333; color: white; }
    .verdict-real { color: #28a745; font-weight: bold; font-size: 40px; text-align: center; }
    .verdict-fake { color: #dc3545; font-weight: bold; font-size: 40px; text-align: center; }

    /* Sidebar Weights Box Styling */
    .weights-box {
        background-color: #e0f2fe; /* Light Blue Bg */
        border: 1px solid #7dd3fc;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .weights-title {
        color: #0369a1;
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 10px;
        display: block;
    }
    .weight-value {
        font-size: 1.2rem;
        font-weight: 600;
        color: #0c4a6e;
        display: block;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADING (The Clean Version)
# ==========================================
@st.cache_resource
def load_models():
    # --- A. Load Text Model (.h5) ---
    try:
        # We use tf_keras to load the legacy .h5 file
        # We must tell it what 'TFBertModel' is using custom_objects
        text_model = tf_keras.models.load_model(
            'final_text_model.h5',
            custom_objects={'TFBertModel': TFBertModel}
        )
        print("Text Model Loaded (Full Model)")
    except Exception as e:
        st.error(f"❌ Text Model Error: {e}")
        text_model = None

    # --- B. Load Image Model (.keras) ---
    try:
        # We use keras (Keras 3) to load the new .keras file
        image_model = keras.models.load_model('final_image_model.keras')
        print("Image Model Loaded (Full Model)")
    except Exception as e:
        st.error(f"❌ Image Model Error: {e}")
        image_model = None

    return text_model, image_model

# Load Resources
text_model, image_model = load_models()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# ==========================================
# 3. SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2964/2964063.png", width=100)
    st.title("Settings")
    st.markdown("### Fusion Balance")
    
    # Single slider controls both weights
    # Value 0.0 = 100% Image, 1.0 = 100% Text
    balance = st.slider("Text vs. Image Importance", 0.0, 1.0, 0.6, 0.05)
    
    # Calculate weights automatically so they ALWAYS sum to 1.0
    w_text = balance
    w_image = 1.0 - balance
    
    # --- CUSTOM CENTERED WEIGHTS BLOCK ---
    st.markdown(f"""
    <div class="weights-box">
        <span class="weights-title">Current Weights</span>
        <span class="weight-value">📝 Text: {w_text:.2f}</span>
        <span class="weight-value">🖼️ Image: {w_image:.2f}</span>
    </div>
    """, unsafe_allow_html=True)
    # -------------------------------------

    st.caption("Architecture: BERT + ResNet50")

# ==========================================
# 4. MAIN UI
# ==========================================
st.title("🕵️‍♀️ Veritas AI | Multimodal Fake News Detector")
st.markdown("##### Detect misinformation using **BERT** (Text) and **ResNet50** (Image).")
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Enter News Details")
    headline = st.text_area("News Headline", placeholder="e.g., Aliens land in New York City...", height=100)
    uploaded_file = st.file_uploader("Upload News Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
    verify_btn = st.button("🔍 Verify Authenticity")

with col2:
    st.subheader("2. Analysis Report")
    
    if verify_btn:
        if not headline or not uploaded_file:
            st.error("⚠️ Please provide both a headline and an image.")
        elif text_model is None or image_model is None:
            st.error("⚠️ Models failed to load. Check console.")
        else:
            with st.spinner("Processing with BERT & ResNet50..."):
                # --- PREDICTION LOGIC ---
                
                # A. Text (BERT) -> Outputs Probability of REAL (Class 1)
                encodings = tokenizer([headline], truncation=True, padding='max_length', 
                                      max_length=64, return_tensors='tf')
                
                # Raw Score (0=Fake, 1=Real)
                p_text_real = text_model.predict({
                    'input_ids': encodings['input_ids'], 
                    'attention_mask': encodings['attention_mask']
                }, verbose=0)[0][0]
                
                # B. Image (ResNet50) -> Outputs Probability of REAL (Class 1)
                img = image.resize((160, 160)) 
                img_array = np.array(img)
                if img_array.shape[-1] == 4: img_array = img_array[..., :3]
                
                img_array = tf_keras.applications.resnet50.preprocess_input(img_array) 
                img_array = tf.expand_dims(img_array, 0)
                
                img_preds = image_model.predict(img_array, verbose=0)[0]
                p_image_real = img_preds[1] # Class 1 is Real
                
                # C. Fusion (Weighted Average of Real Scores)
                p_final_real = (w_text * p_text_real) + (w_image * p_image_real)
                
                # D. Final Decision
                # If Score > 0.5, it is REAL.
                if p_final_real > 0.5:
                    label = "REAL NEWS"
                    # Confidence is the score itself (e.g., 0.99)
                    confidence = p_final_real
                    st.markdown(f'<div class="verdict-real">✅ {label}</div>', unsafe_allow_html=True)
                else:
                    label = "FAKE NEWS"
                    # Confidence is the distance from 0 (e.g., 1 - 0.05 = 0.95)
                    confidence = 1.0 - p_final_real
                    st.markdown(f'<div class="verdict-fake">🚨 {label}</div>', unsafe_allow_html=True)
                
                st.progress(float(confidence), text=f"Confidence: {confidence:.1%}")
                st.markdown("---")
                
                # E. Breakdown (SHOWING FAKE PROBABILITY)
                # To see "Fake Prob", we do (1.0 - Real Score)
                c1, c2 = st.columns(2)
                with c1:
                    fake_prob_text = 1.0 - p_text_real
                    st.metric("BERT Fake Probability", f"{fake_prob_text:.2%}")
                with c2:
                    fake_prob_image = 1.0 - p_image_real
                    st.metric("ResNet Fake Probability", f"{fake_prob_image:.2%}")

    else:
        st.info("Waiting for input...")