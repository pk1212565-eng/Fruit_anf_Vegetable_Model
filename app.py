import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image, ImageOps
import os
import json
from scipy.signal import convolve2d

# --- 1. PREMIUM PRODUCTION PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FreshScan — Advanced Deep Learning Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. LIGHT & CLEAN INDUSTRIAL THEME (CUSTOM CSS) ---
st.markdown("""
    <style>
    /* Global background adjust */
    .main { background-color: #f8fafc; }
    
    /* Clean layout cards */
    .premium-card {
        background-color: #ffffff;
        padding: 26px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px -2px rgba(15, 23, 42, 0.04);
        margin-bottom: 24px;
    }
    
    /* Typography customization */
    h1, h2, h3, h4, h5 {
        color: #0f172a !important;
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 700 !important;
    }
    
    /* Elegant terminal-style codeblocks */
    code {
        color: #0f172a !important;
        background-color: #f1f5f9 !important;
        font-size: 0.92em !important;
        border-radius: 8px !important;
        padding: 4px 8px !important;
    }
    
    /* Smooth Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 10px; background-color: #e2e8f0; padding: 6px; border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 46px; background-color: transparent; border-radius: 8px;
        color: #475569; font-weight: 600; padding: 0px 20px; border: none !important;
        transition: all 0.2s ease-in-out;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #ffffff !important; color: #0284c7 !important;
        box-shadow: 0 2px 6px 0 rgba(0, 0, 0, 0.08);
    }
    
    /* Action Button Enhancement */
    div.stButton > button:first-child {
        background-color: #0284c7; color: white; border-radius: 8px;
        font-weight: 600; padding: 10px 24px; border: none; transition: 0.2s;
    }
    div.stButton > button:first-child:hover { background-color: #0369a1; }
    </style>
""", unsafe_allow_html=True)

# --- 3. BACKEND BACKBONE MECHANICS (MODEL & LABELS) ---
@st.cache_resource
def load_production_model():
    model_file = "Fruite_and_veg_model.keras"
    if os.path.exists(model_file):
        try:
            return tf.keras.models.load_model(model_file)
        except Exception as e:
            st.error(f"Backend layout engine loading error: {e}")
            return None
    return None

model = load_production_model()

def load_target_labels():
    if os.path.exists("class_names.json"):
        try:
            with open("class_names.json", "r") as f:
                return json.load(f)
        except:
            pass
    # Fallback to standard 36 classes commonly used in Kaggle fruit/veg datasets
    return [
        "Apple", "Banana", "Beetroot", "Cabbage", "Capsicum", "Carrot", "Cauliflower",
        "Chilli Pepper", "Corn", "Cucumber", "Eggplant", "Garlic", "Ginger", "Grapes",
        "Jalapeno", "Kiwi", "Lemon", "Lettuce", "Mango", "Onion", "Orange", "Paprika",
        "Pear", "Peas", "Pineapple", "Pomegranate", "Potato", "Raddish", "Soy Beans",
        "Spinach", "Sweetcorn", "Sweetpotato", "Tomato", "Turnip", "Watermelon"
    ]

class_names = load_target_labels()

# --- 4. HEADER CONTROL PANEL ---
st.title("🧠 FreshScan — Deep Learning & CV Studio")
st.markdown("A premium production-ready dashboard engineered to simulate core neural computations and evaluate live Keras vision pipelines.")
st.markdown("---")

# --- 5. FIXED SIDEBAR INPUT MECHANICS ---
with st.sidebar:
    st.markdown('### 🎯 System Infrastructure', unsafe_allow_html=True)
    if model is not None:
        st.success("🟢 Keras Core Architecture: Active")
    else:
        st.warning("🟡 Waiting for Fruite_and_veg_model.keras via Git LFS")
    
    st.markdown("---")
    st.markdown("##### 🎛️ Perceptron Flow Tuning")
    x1 = st.slider("Input Scalar $x_1$", -2.0, 2.0, 1.0, 0.1)
    w1 = st.slider("Weight Vector $w_1$", -2.0, 2.0, 0.5, 0.1)
    x2 = st.slider("Input Scalar $x_2$", -2.0, 2.0, -0.5, 0.1)
    w2 = st.slider("Weight Vector $w_2$", -2.0, 2.0, 1.2, 0.1)
    b = st.slider("Scalar Bias $b$", -2.0, 2.0, -0.2, 0.1)
    
    st.markdown("---")
    st.caption("FreshScan Framework v2.4 (Optimized)")

# --- 6. CORE ARCHITECTURE WORKSPACE ---
tabs = st.tabs([
    "⚡ Perceptron Flow Optimization", 
    "📊 Non-Linearity Analytics", 
    "🔍 Real-Time Computer Vision Mapping", 
    "🚀 Live Scan Production Inference"
])

# ==========================================
# TAB 1: PERCEPTRON FLOW OPTIMIZATION
# ==========================================
with tabs[0]:
    st.subheader("Forward Network Matrix Flow")
    col1, col2 = st.columns([1, 1.3], gap="large")
    
    with col1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("##### 📈 Forward Array Activations")
        z = x1*w1 + x2*w2 + b
        a = 1 / (1 + np.exp(-z))
        loss = 0.5 * (a - 1.0)**2 # Target defaulting to 1.0 (binary ground truth mapping)
        
        st.code(f"Linear Combination (z) = {z:.4f}\nSigmoid Map Activation (a) = {a:.4f}\nMean Squared Error Loss = {loss:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("##### 📉 Gradient Backpropagation Nodes")
        dL_da = a - 1.0
        da_dz = a * (1 - a)
        dL_dz = dL_da * da_dz
        
        st.markdown("Partial derivatives computed via Chain Rule mapping:")
        st.latex(rf"\frac{{\partial L}}{{\partial w_1}} = {dL_dz*x1:.4f} \quad | \quad \frac{{\partial L}}{{\partial w_2}} = {dL_dz*x2:.4f} \quad | \quad \frac{{\partial L}}{{\partial b}} = {dL_dz:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 2: NON-LINEARITY ANALYTICS
# ==========================================
with tabs[1]:
    st.subheader("Activation Profiles Configuration")
    col_act1, col_act2 = st.columns([1, 1.5], gap="large")
    
    with col_act1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        act_profile = st.selectbox("Select Network Function Profile", ["ReLU", "Sigmoid", "Tanh", "LeakyReLU"])
        st.markdown("---")
        if act_profile == "ReLU": 
            st.latex(r"f(x) = \max(0, x)")
            st.caption("Solves vanishing gradient issues during multi-layer adjustments.")
        elif act_profile == "Sigmoid": 
            st.latex(r"f(x) = \frac{1}{1 + e^{-x}}")
            st.caption("Compresses raw activations into clean probability distributions.")
        elif act_profile == "Tanh": 
            st.latex(r"f(x) = \tanh(x)")
            st.caption("Zero-centered output space profiles accelerate deep stack training sequences.")
        elif act_profile == "LeakyReLU": 
            st.latex(r"f(x) = \max(0.1x, x)")
            st.caption("Prevents complete gradient cell death by introducing small trailing alpha scales.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_act2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        x_space = np.linspace(-5, 5, 200)
        if act_profile == "ReLU": y_space = np.maximum(0, x_space)
        elif act_profile == "Sigmoid": y_space = 1 / (1 + np.exp(-x_space))
        elif act_profile == "Tanh": y_space = np.tanh(x_space)
        elif act_profile == "LeakyReLU": y_space = np.where(x_space > 0, x_space, 0.1 * x_space)
        
        fig, ax = plt.subplots(figsize=(6.5, 2.6))
        ax.plot(x_space, y_space, color='#0284c7', lw=2.5)
        ax.grid(True, linestyle=':', alpha=0.5, color='#cbd5e1')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 3: REAL-TIME COMPUTER VISION MAPPING
# ==========================================
with tabs[2]:
    st.subheader("Interactive 2D Convolution Engine")
    st.markdown("Upload any external evaluation image below to witness how multi-layered structural filter mappings extract geometric edge contours in raw 2D pixel grids.")
    
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    cv_upload = st.file_uploader("Bring image file from your system workspace", type=["jpg", "jpeg", "png"], key="cv_pipeline_uploader")
    
    if cv_upload is not None:
        grid_1, grid_2, grid_3 = st.columns(3)
        
        raw_img = Image.open(cv_upload)
        with grid_1:
            st.markdown("##### 🖼️ Source Canvas Input")
            st.image(raw_img, use_container_width=True)
            
        gray_canvas = ImageOps.grayscale(raw_img).resize((350, 350))
        pixel_array = np.array(gray_canvas)
        with grid_2:
            st.markdown("##### 🔲 Grayscale Structural View")
            st.image(gray_canvas, use_container_width=True)
            
        # Defining standard Sobel Laplacian high-pass filtration array
        laplacian_filter = np.array([[-1, -1, -1], 
                                     [-1,  8, -1], 
                                     [-1, -1, -1]])
        
        # Array convolution computation sequence
        feature_map = convolve2d(pixel_array, laplacian_filter, mode='same')
        feature_map = np.clip(feature_map, 0, 255).astype(np.uint8)
        
        with grid_3:
            st.markdown("##### ⚡ Output Feature Activation Map")
            st.image(feature_map, use_container_width=True, channels="L")
            st.success("Matrix 2D spatial extraction map compiled successfully!")
    else:
      st.info("💡 Drop an image inside the workspace above to unlock live matrix pixel arrays and edge contour visualizations.")
      st.markdown('', unsafe_allow_html=True)

==========================================


TAB 4: LIVE SCAN PRODUCTION INFERENCE


==========================================

with tabs[3]:
st.subheader("Keras Core Target Inference")
if model is None:
st.info("💡 Evaluation pipeline offline. To serve active inferences, push your 'Fruite_and_veg_model.keras' array using Git LFS setup commands into your cloud branch repository.")
else:
st.markdown('', unsafe_allow_html=True)
prod_upload = st.file_uploader("Upload target sample matrix for class processing", type=["jpg", "jpeg", "png"], key="inference_system_uploader")
if prod_upload is not None:
col_inf1, col_inf2 = st.columns([1, 1.2], gap="medium")
target_image = Image.open(prod_upload)
with col_inf1:
st.image(target_image, caption="Inference Object Buffer", width=280)
# Preprocessing configuration parameters matching standard image batch pipelines
resized_matrix = target_image.resize((180, 180))
input_tensor = np.array(resized_matrix)
# Normalization and dimensionality expands check
if input_tensor.shape[-1] == 4:  # Strip transparency matrix channel if present
input_tensor = input_tensor[..., :3]
input_tensor = input_tensor / 255.0
input_tensor = np.expand_dims(input_tensor, axis=0)
with col_inf2:
if st.button("Execute Model Evaluation", use_container_width=True):
with st.spinner("Processing structural network nodes array..."):
raw_logits = model.predict(input_tensor)
# Dynamic softmax activation layer computation array
softmax_activation = np.exp(raw_logits - np.max(raw_logits))
distribution_probabilities = softmax_activation / softmax_activation.sum()
max_target_index = np.argmax(distribution_probabilities)
confidence_score = distribution_probabilities[0][max_target_index] * 100
st.markdown("### 🏆 Prediction Diagnostics Summary")
st.metric(label="Detected Structural Target", value=str(class_names[max_target_index]))
st.metric(label="Model Backbone Confidence Rating", value=f"{confidence_score:.2f}%")
st.markdown('', unsafe_allow_html=True)
