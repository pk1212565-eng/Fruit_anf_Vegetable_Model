import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image, ImageOps
import os
import json
from scipy.signal import convolve2d

# --- 1. INDUSTRIAL CLEAN PAGE CONFIGURATION ---
st.set_page_config(
    page_title="VisionStudio — Deep Learning Core",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. PREMIUM MODERN MINIMALIST VISUAL THEME (CUSTOM CSS) ---
st.markdown("""
    <style>
    /* Premium background grid styling */
    .main { background-color: #fafafa; }
    
    /* Elegant rounded studio panels */
    .studio-card {
        background-color: #ffffff;
        padding: 28px;
        border-radius: 16px;
        border: 1px solid #f0f0f2;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.02), 0 8px 10px -6px rgba(0, 0, 0, 0.02);
        margin-bottom: 24px;
    }
    
    /* Professional sleek headers */
    h1, h2, h3, h4, h5 {
        color: #111827 !important;
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    /* Clean navigation tabs customization */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 12px; background-color: #f3f4f6; padding: 6px; border-radius: 14px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px; background-color: transparent; border-radius: 10px;
        color: #4b5563; font-weight: 600; padding: 0px 24px; border: none !important;
        transition: all 0.25s ease;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #ffffff !important; color: #0284c7 !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    }
    
    /* Modern unified buttons styling */
    div.stButton > button:first-child {
        background-color: #111827; color: white; border-radius: 10px;
        font-weight: 600; padding: 12px 28px; border: none; transition: 0.2s ease;
    }
    div.stButton > button:first-child:hover { background-color: #0284c7; }
    </style>
""", unsafe_allow_html=True)

# --- 3. PRODUCTION BACKEND GRAPH ENGINE ---
@st.cache_resource
def load_vision_model():
    model_file = "Fruite_and_veg_model.keras"
    if os.path.exists(model_file):
        try:
            return tf.keras.models.load_model(model_file)
        except:
            return None
    return None

model = load_vision_model()

def load_studio_labels():
    if os.path.exists("class_names.json"):
        try:
            with open("class_names.json", "r") as f:
                return json.load(f)
        except:
            pass
    return [
        "Apple", "Banana", "Beetroot", "Cabbage", "Capsicum", "Carrot", "Cauliflower",
        "Chilli Pepper", "Corn", "Cucumber", "Eggplant", "Garlic", "Ginger", "Grapes",
        "Jalapeno", "Kiwi", "Lemon", "Lettuce", "Mango", "Onion", "Orange", "Paprika",
        "Pear", "Peas", "Pineapple", "Pomegranate", "Potato", "Raddish", "Soy Beans",
        "Spinach", "Sweetcorn", "Sweetpotato", "Tomato", "Turnip", "Watermelon"
    ]

class_names = load_studio_labels()

# --- 4. GRAPHICAL CONTROL MONITOR HEADER ---
st.title("👁️ VisionStudio — Deep Learning Lab")
st.markdown("A purely visual, non-mathematical interaction lab tracking feature maps processing across multi-layer convolutional nodes.")
st.markdown("---")

# --- 5. MINIMALIST SIDEBAR TELEMETRY ---
with st.sidebar:
    st.markdown('### 🖥️ Network Core Node', unsafe_allow_html=True)
    if model is not None:
        st.success("🟢 Model Architecture Connected")
    else:
        st.warning("⚠️ Local weights array offline")
    st.markdown("---")
    st.info("💡 Adjust node properties using tabs in your central dashboard canvas.")

# --- 6. CORE ARCHITECTURE LIVE CANVAS ---
tabs = st.tabs([
    "🧠 Neural Network Fundamentals", 
    "📊 Non-Linearity & Errors", 
    "🔍 CNN Spatial Feature Extraction", 
    "🚀 Live Scan Inference Studio"
])

# ==========================================
# TAB 1: NEURAL NETWORK FUNDAMENTALS
# ==========================================
with tabs[0]:
    st.subheader("Information Flow In Neural Architecture")
    st.markdown("Understand how data transitions across a model layer nodes array from raw values to class targets.")
    
    col_flow1, col_flow2 = st.columns(2, gap="large")
    
    with col_flow1:
        st.markdown('<div class="studio-card">', unsafe_allow_html=True)
        st.markdown("##### ⚡ Forward Propagation Path")
        st.markdown("""
        * **Input Layer Node:** Raw feature signals enter the node blocks array.
        * **Synaptic Weight Scaling:** Signals are amplified or dampened based on connecting layer importance values.
        * **Bias Offset Node:** Shifting vector ranges to adjust internal triggering scales.
        * **Activation Gateway:** Encodes features to determine if a specific neuron node should fire.
        """)
        
        # Purely graphical representation of a neural node forward path
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.text(0.1, 0.5, 'Inputs\n(Data)', ha='center', va='center', bbox=dict(boxstyle='round', facecolor='#f3f4f6', edgecolor='#cbd5e1'))
        ax.text(0.5, 0.5, 'Perceptron Node\n(Weight Matrix)', ha='center', va='center', bbox=dict(boxstyle='circle', facecolor='#e0f2fe', edgecolor='#0284c7'))
        ax.text(0.9, 0.5, 'Output\n(Prediction)', ha='center', va='center', bbox=dict(boxstyle='round', facecolor='#dcfce7', edgecolor='#22c55e'))
        ax.annotate('', xy=(0.33, 0.5), xytext=(0.18, 0.5), arrowprops=dict(arrowstyle="->", color='#94a3b8', lw=2))
        ax.annotate('', xy=(0.77, 0.5), xytext=(0.67, 0.5), arrowprops=dict(arrowstyle="->", color='#0284c7', lw=2))
        ax.axis('off')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_flow2:
        st.markdown('<div class="studio-card">', unsafe_allow_html=True)
        st.markdown("##### 🔄 Backpropagation & Learning Optimization")
        st.markdown("""
        * **Error Evaluation:** System measures deviation between active inference layers and ground labels.
        * **Reverse Gradient Tracking:** Feedback loops drive errors backwards into early layer matrices.
        * **Weight Configuration Tweak:** Network layers adjust internal nodes structure until error delta converges.
        """)
        
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.text(0.1, 0.5, 'Optimized\nWeights', ha='center', va='center', bbox=dict(boxstyle='round', facecolor='#fafafa', edgecolor='#e2e8f0'))
        ax.text(0.5, 0.5, 'Gradient Update\nEngine', ha='center', va='center', bbox=dict(boxstyle='round', facecolor='#f3e8ff', edgecolor='#a855f7'))
        ax.text(0.9, 0.5, 'Loss Delta\n(Error Scale)', ha='center', va='center', bbox=dict(boxstyle='round', facecolor='#fee2e2', edgecolor='#ef4444'))
        ax.annotate('', xy=(0.22, 0.5), xytext=(0.38, 0.5), arrowprops=dict(arrowstyle="->", color='#a855f7', lw=2))
        ax.annotate('', xy=(0.62, 0.5), xytext=(0.78, 0.5), arrowprops=dict(arrowstyle="->", color='#ef4444', lw=2))
        ax.axis('off')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 2: NON-LINEARITY & ERRORS
# ==========================================
with tabs[1]:
    st.subheader("Activation & Model Optimization Curves")
    col_an1, col_an2 = st.columns([1, 1.4], gap="large")
    
    with col_an1:
        st.markdown('<div class="studio-card">', unsafe_allow_html=True)
        function_profile = st.selectbox("Select Layer Gate Profile", ["ReLU Profile", "Sigmoid Gate", "Tanh Range"])
        st.markdown("---")
        if "ReLU" in function_profile:
            st.markdown("**ReLU Feature Handling:** Blocks any feature properties falling into negative scale arrays, rendering sparse activation clusters.")
        elif "Sigmoid" in function_profile:
            st.markdown("**Sigmoid Strategy:** Normalizes spatial activation vector limits to output values tracking probabilistic distributions.")
        elif "Tanh" in function_profile:
            st.markdown("**Tanh Distribution:** Zero-centered scaling engine allowing fast balancing optimizations within heavy deep networks.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_an2:
        st.markdown('<div class="studio-card">', unsafe_allow_html=True)
        x_graph = np.linspace(-4, 4, 150)
        if "ReLU" in function_profile: y_graph = np.maximum(0, x_graph)
        elif "Sigmoid" in function_profile: y_graph = 1 / (1 + np.exp(-x_graph))
        elif "Tanh" in function_profile: y_graph = np.tanh(x_graph)
        
        fig, ax = plt.subplots(figsize=(6, 2.4))
        ax.plot(x_graph, y_graph, color='#0284c7', lw=3)
        ax.axhline(0, color='#e5e7eb', lw=1.2, linestyle='--')
        ax.axvline(0, color='#e5e7eb', lw=1.2, linestyle='--')
        ax.axis('on')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 3: CNN SPATIAL FEATURE EXTRACTION
# ==========================================
with tabs[2]:
    st.subheader("Convolutional Network Feature Mapping")
    st.markdown("Upload any validation image to view how CNN filter matrices scan structural profiles, segmenting raw objects into clean geometric lines.")
    
    st.markdown('<div class="studio-card">', unsafe_allow_html=True)
    image_input_cv = st.file_uploader("Upload an evaluation image file", type=["jpg", "jpeg", "png"], key="studio_cv_uploader")
    
    if image_input_cv is not None:
        cv_col1, cv_col2 = st.columns(2, gap="large")
        
        base_canvas = Image.open(image_input_cv)
        with cv_col1:
            st.markdown("##### 🖼️ Source Canvas Input")
            st.image(base_canvas, use_container_width=True)
            
