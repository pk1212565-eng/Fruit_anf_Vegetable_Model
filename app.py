import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image, ImageOps
import os
import json
from scipy.signal import convolve2d

# --- PREMIUM PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FreshScan — Fruits & Vegetables Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN THEME CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #fcfcfd; }
    .premium-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #eef2f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    h1, h2, h3, h5 {
        color: #1e293b !important;
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 700 !important;
    }
    code {
        color: #0f172a !important;
        background-color: #f1f5f9 !important;
        font-size: 0.9em !important;
        border-radius: 6px !important;
    }
    .stTabs [data-baseweb="tab-list"] { 
        gap: 8px; background-color: #f1f5f9; padding: 6px; border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px; background-color: transparent; border-radius: 6px;
        color: #64748b; font-weight: 600; padding: 0px 16px; border: none !important;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #ffffff !important; color: #0f172a !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    .sb-header {
        font-size: 1.1rem; font-weight: bold; color: #475569;
        margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD DEPLOYED KERAS MODEL ---
@st.cache_resource
def load_keras_model():
    model_file = "Fruite_and_veg_model.keras"
    if os.path.exists(model_file):
        try:
            return tf.keras.models.load_model(model_file)
        except Exception as e:
            st.error(f"Error loading model layout: {e}")
            return None
    return None

model = load_keras_model()

# --- LOAD CLASS LABELS ---
def load_labels():
    if os.path.exists("class_names.json"):
        with open("class_names.json", "r") as f:
            return json.load(f)
    return ["Apple", "Banana", "Orange", "Tomato", "Potato", "Carrot"]

class_names = load_labels()

# --- HEADER SECTION ---
st.title("🧠 FreshScan — Fruits & Vegetables Studio")
st.markdown("Interactive workspace designed to experiment with neural networks and analyze custom Keras classification predictions.")
st.markdown("---")

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.markdown('<p class="sb-header">🎯 Workspace Status</p>', unsafe_allow_html=True)
    if model is not None:
        st.success("🟢 Keras Backbone Active")
    else:
        st.warning("🟡 Fruite_and_veg_model.keras not detected locally")
    
    st.markdown("---")
    st.markdown("##### ⚙️ Perceptron Configuration")
    x1 = st.slider("Input $x_1$", -2.0, 2.0, 1.0, 0.1)
    w1 = st.slider("Weight $w_1$", -2.0, 2.0, 0.5, 0.1)
    x2 = st.slider("Input $x_2$", -2.0, 2.0, -0.5, 0.1)
    w2 = st.slider("Weight $w_2$", -2.0, 2.0, 1.2, 0.1)
    b = st.slider("Bias $b$", -2.0, 2.0, -0.2, 0.1)
    
    st.markdown("---")
    st.caption("FreshScan Web Architecture Layout Panel")

# --- NAVIGATION TABS ---
tabs = st.tabs([
    "⚡ Perceptron & Propagation", 
    "📊 Activation & Loss Functions", 
    "🔍 Convolutional Operations", 
    "🚀 Live Scanner Model Inference"
])

# ==========================================
# TAB 1: PERCEPTRON & PROPAGATION
# ==========================================
with tabs[0]:
    st.subheader("Perceptron & Information Flow")
    col1, col2 = st.columns([1, 1.5], gap="large")
    
    with col1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("##### 📈 Forward Pass Calculations")
        z = x1*w1 + x2*w2 + b
        a = 1 / (1 + np.exp(-z))
        loss = 0.5 * (a - 1.0)**2
        
        st.code(f"Sum (z) = {z:.4f}\nActivation (a) = {a:.4f}\nLoss (MSE) = {loss:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("##### 📉 Gradient Backpropagation Updates")
        dL_da = a - 1.0
        da_dz = a * (1 - a)
        dL_dz = dL_da * da_dz
        
        st.latex(rf"\frac{{\partial L}}{{\partial w_1}} = {dL_dz*x1:.4f} \quad | \quad \frac{{\partial L}}{{\partial w_2}} = {dL_dz*x2:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 2: ACTIVATION & LOSS FUNCTIONS
# ==========================================
with tabs[1]:
    st.subheader("Activation Math Analytics")
    col_act1, col_act2 = st.columns([1, 2], gap="large")
    
    with col_act1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        act_choice = st.selectbox("Activation Profile", ["ReLU", "Sigmoid", "Tanh", "LeakyReLU"])
        if act_choice == "ReLU": st.latex(r"f(x) = \max(0, x)")
        elif act_choice == "Sigmoid": st.latex(r"f(x) = \frac{1}{1 + e^{-x}}")
        elif act_choice == "Tanh": st.latex(r"f(x) = \tanh(x)")
        elif act_choice == "LeakyReLU": st.latex(r"f(x) = \max(0.1x, x)")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_act2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        x_val = np.linspace(-5, 5, 100)
        if act_choice == "ReLU": y_val = np.maximum(0, x_val)
        elif act_choice == "Sigmoid": y_val = 1 / (1 + np.exp(-x_val))
        elif act_choice == "Tanh": y_val = np.tanh(x_val)
        elif act_choice == "LeakyReLU": y_val = np.where(x_val > 0, x_val, 0.1 * x_val)
        
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.plot(x_val, y_val, color='#2563eb', lw=2)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 3: CONVOLUTIONAL OPERATIONS (UPLOAD ADDED!)
# ==========================================
with tabs[2]:
    st.subheader("Matrix Convolution & Visual Feature Extraction")
    st.markdown("Niche diye gaye section mein aap apne computer se koi bhi image upload karke dekh sakte hain ke Convolution Filter edges ko kaise extract karta hai.")
    
    # 📸 REAL IMAGE UPLOADER FOR CNN VISUALIZATION
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("##### 📤 Upload an Image for Convolution Feature Map Extraction")
    cnn_uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="cnn_uploader")
    
    if cnn_uploaded_file is not None:
        c_col1, c_col2, c_col3 = st.columns(3)
        
        # 1. Original Image display
        orig_img = Image.open(cnn_uploaded_file)
        with c_col1:
            st.markdown("**Original Uploaded Image:**")
            st.image(orig_img, use_container_width=True)
            
        # 2. Convert to Grayscale (Matrix View)
        gray_img = ImageOps.grayscale(orig_img).resize((300, 300))
        img_matrix = np.array(gray_img)
        with c_col2:
            st.markdown("**Grayscale Matrix View (300x300):**")
            st.image(gray_img, use_container_width=True)
            
        # 3. Apply Edge Detection Filter (Convolution)
        # Sobel/Edge Filter Definition
        edge_kernel = np.array([[-1, -1, -1], 
                                [-1,  8, -1], 
                                [-1, -1, -1]])
        
        # Run 2D Convolution on the real image
        conv_features = convolve2d(img_matrix, edge_kernel, mode='same')
        conv_features = np.clip(conv_features, 0, 255).astype(np.uint8)
        
        with c_col3:
            st.markdown("**CNN Output Feature Map (Detected Edges):**")
            st.image(conv_features, use_container_width=True, channels="L")
            st.success("Convolution operation completed on your image!")
    st.markdown('</div>', unsafe_allow_html=True)

    # Dummy Matrix Math Simulation (Keep previous feature)
    st.markdown("##### 🧮 Math Dummy Matrix Simulation")
    input_matrix = np.array([[1, 1, 1, 0, 0],[0, 1, 1, 1, 0],[0, 0, 1, 1, 1],[0, 0, 1, 1, 0],[0, 1, 1, 0, 0]])
    kernel = np.array([[1, 0, 1],[0, 1, 0],[1, 0, 1]])
    
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("##### 🖼️ 5x5 Matrix Input")
        st.dataframe(input_matrix, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_c2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("##### 🔍 3x3 Filter")
        st.dataframe(kernel, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_c3:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("##### ⚡ Output Feature Map")
        if st.button("Trigger 2D Convolution Simulation", use_container_width=True):
            out = np.zeros((3,3))
            for i in range(3):
                for j in range(3):
                    out[i,j] = np.sum(input_matrix[i:i+3, j:j+3] * kernel)
            st.dataframe(out.astype(int), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 4: LIVE SCANNER MODEL INFERENCE (PREDICTION)
# ==========================================
with tabs:
    st.subheader("Live Scanner Inference Studio")
    if model is None:
        st.info("💡 To analyze active production pipelines, ensure 'Fruite_and_veg_model.keras' is fully pushed via Git LFS into your working branch folder.")
    else:
        st.markdown('', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Fruit/Vegetable Image for Prediction", type=["jpg", "jpeg", "png"], key="predict_uploader")
if uploaded_file is not None:
img = Image.open(uploaded_file)
st.image(img, caption="Target Scan Subject", width=250)
# Preprocessing matching notebook config
img_resized = img.resize((180, 180))
img_array = np.array(img_resized) / 255.0
img_array = np.expand_dims(img_array, axis=0)
if st.button("Run Model Prediction", use_container_width=True):
with st.spinner("Executing dense array classification..."):
predictions = model.predict(img_array)
exp_preds = np.exp(predictions - np.max(predictions))
probabilities = exp_preds / exp_preds.sum()
top_idx = np.argmax(probabilities)
st.success(f"🎯 Predicted Target: {class_names[top_idx]} ({probabilities[top_idx]*100:.2f}%)")
st.markdown('', unsafe_allow_html=True)
