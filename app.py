import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# --- PREMIUM PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Neural Networks & Computer Vision Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN LIGHT THEME CUSTOM CSS ---
st.markdown("""
    <style>
    /* Global body styling */
    .main { 
        background-color: #fcfcfd; 
    }
    
    /* Modern minimalist Card containers */
    .premium-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #eef2f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
    }
    
    /* Sleek Typography */
    h1, h2, h3 {
        color: #1e293b !important;
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 700 !important;
    }
    
    /* Custom style for clean code and outputs */
    code {
        color: #0f172a !important;
        background-color: #f1f5f9 !important;
        font-size: 0.9em !important;
        border-radius: 6px !important;
    }
    
    /* Tab Styling Customization */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 8px;
        background-color: #f1f5f9;
        padding: 6px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre;
        background-color: transparent;
        border-radius: 6px;
        color: #64748b;
        font-weight: 600;
        padding: 0px 16px;
        border: none !important;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #ffffff !important; 
        color: #0f172a !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    /* Accent Sidebar Header */
    .sb-header {
        font-size: 1.1rem;
        font-weight: bold;
        color: #475569;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.title("🧠 Deep Learning & CV Studio")
st.markdown("An interactive workspace designed to experiment with neural network architectures, flow mechanics, and computer vision operations.")
st.markdown("---")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.markdown('<p class="sb-header">🎯 Global Settings</p>', unsafe_allow_html=True)
    st.info("💡 Adjust network inputs or switch architecture modules below.")
    st.markdown("---")
    st.caption("Developed for production-ready deployment.")

# --- NAVIGATION TABS ---
tabs = st.tabs([
    "⚡ Perceptron & Propagation", 
    "📊 Activation & Loss Functions", 
    "🔍 Convolutional Operations", 
    "🚀 Keras Transfer Learning"
])

# ==========================================
# TAB 1: PERCEPTRON & PROPAGATION
# ==========================================
with tabs[0]:
    st.subheader("Perceptron & Information Flow")
    st.markdown("Observe forward computations and automatic gradient updates in real time based on your configuration.")
    
    col1, col2 = st.columns([1, 1.5], gap="large")
    
    with col1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("##### 🎛️ Parameters Configuration")
        
        # Sub-grid for input clean alignment
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            x1 = st.slider("Input $x_1$", -2.0, 2.0, 1.0, 0.1)
            w1 = st.slider("Weight $w_1$", -2.0, 2.0, 0.5, 0.1)
        with sub_col2:
            x2 = st.slider("Input $x_2$", -2.0, 2.0, -0.5, 0.1)
            w2 = st.slider("Weight $w_2$", -2.0, 2.0, 1.2, 0.1)
            
        b = st.slider("Bias $b$", -2.0, 2.0, -0.2, 0.1)
        
        sub_col3, sub_col4 = st.columns(2)
        with sub_col3:
            target = st.number_input("Target ($y$)", value=1.0)
        with sub_col4:
            lr = st.slider("Learning Rate ($\eta$)", 0.01, 1.0, 0.1)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Calculations
        z = x1*w1 + x2*w2 + b
        a = 1 / (1 + np.exp(-z))
        loss = 0.5 * (a - target)**2
        
    with col2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("##### 📈 Forward & Backward Execution Pipeline")
        
        st.markdown("**Forward Pass Outputs:**")
        st.code(f"Sum (z) = {z:.4f}\nActivation (a) = {a:.4f}\nMean Squared Error Loss = {loss:.4f}")
        
        # Gradients Calculation
        dL_da = a - target
        da_dz = a * (1 - a)
        dL_dz = dL_da * da_dz
        
        dL_dw1, dL_dw2, dL_db = dL_dz * x1, dL_dz * x2, dL_dz
        
        st.markdown("**Computed Gradients ($\partial L / \partial w$):**")
        st.latex(rf"\frac{{\partial L}}{{\partial w_1}} = {dL_dw1:.4f} \quad | \quad \frac{{\partial L}}{{\partial w_2}} = {dL_dw2:.4f} \quad | \quad \frac{{\partial L}}{{\partial b}} = {dL_db:.4f}")
        
        # Optimized Weights
        w1_n, w2_n, b_n = w1 - lr * dL_dw1, w2 - lr * dL_dw2, b - lr * dL_db
        
        st.markdown("**Updated Weight Configuration:**")
        st.code(f"w1_updated = {w1_n:.4f}\nw2_updated = {w2_n:.4f}\nb_updated  = {b_n:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 2: ACTIVATION & LOSS FUNCTIONS
# ==========================================
with tabs[1]:
    st.subheader("Activation & Loss Math Analytics")
    
    col_act1, col_act2 = st.columns([1, 2], gap="large")
    
    with col_act1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("##### 📊 Select Function")
        act_choice = st.selectbox("Activation Function Target", ["ReLU", "Sigmoid", "Tanh", "LeakyReLU"])
        
        # Math Explanations 
        if act_choice == "ReLU":
            st.latex(r"f(x) = \max(0, x)")
            st.caption("Eliminates vanishing gradient issues but can cause dead neurons.")
        elif act_choice == "Sigmoid":
            st.latex(r"f(x) = \frac{1}{1 + e^{-x}}")
            st.caption("Outputs probabilities between 0 and 1.")
        elif act_choice == "Tanh":
            st.latex(r"f(x) = \tanh(x)")
            st.caption("Zero-centered output (-1 to 1) making optimization easier.")
        elif act_choice == "LeakyReLU":
            st.latex(r"f(x) = \max(0.1x, x)")
            st.caption("Prevents dying neurons by keeping a small slope.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_act2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        x_val = np.linspace(-5, 5, 200)
        if act_choice == "ReLU": y_val = np.maximum(0, x_val)
        elif act_choice == "Sigmoid": y_val = 1 / (1 + np.exp(-x_val))
        elif act_choice == "Tanh": y_val = np.tanh(x_val)
        elif act_choice == "LeakyReLU": y_val = np.where(x_val > 0, x_val, 0.1 * x_val)
        
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.plot(x_val, y_val, color='#2563eb', lw=2.5, label=act_choice)
        ax.axhline(0, color='#cbd5e1', lw=1, linestyle='--')
        ax.axvline(0, color='#cbd5e1', lw=1, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#94a3b8')
        ax.spines['bottom'].set_color('#94a3b8')
        ax.tick_params(colors='#64748b', labelsize=9)
        ax.grid(True, linestyle=':', alpha=0.5, color='#cbd5e1')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 3: CONVOLUTIONAL OPERATIONS
# ==========================================
with tabs[2]:
    st.subheader("Matrix Convolution Simulation")
    st.markdown("Analyze step-by-step element-wise matrix dot product matrix mappings.")
    
    input_matrix = np.array([[1, 1, 1, 0, 0],[0, 1, 1, 1, 0],[0, 0, 1, 1, 1],[0, 0, 1, 1, 0],[0, 1, 1, 0, 0]])
    kernel = np.array([[1, 0, 1],[0, 1, 0],[1, 0, 1]])
    
    col_cnn1, col_cnn2, col_cnn3 = st.columns(3)
    
    with col_cnn1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("##### 🖼️ 5x5 Input Image")
        st.dataframe(input_matrix, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_cnn2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("##### 🔍 3x3 Kernel Filter")
        st.dataframe(kernel, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_cnn3:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("##### ⚡ Feature Output Map")
        if st.button("Trigger 2D Convolution", use_container_width=True):
            output_matrix = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    output_matrix[i, j] = np.sum(input_matrix[i:i+3, j:j+3] * kernel)
            st.dataframe(output_matrix.astype(int), use_container_width=True)
            st.success("Convolution complete!")
        else:
            st.info("Click button to compute mapping.")
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 4: TRANSFER LEARNING (KERAS INTEGRATION)
# ==========================================
with tabs[3]:
    st.subheader("Keras Model Integration Studio")
    st.markdown("Manage configurations for models saved locally or using pretrained networks via Keras application blocks.")
    
    col_tl1, col_tl2 = st.columns([1, 1.5], gap="large")
    
    with col_tl1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("##### 📁 Model Settings")
        
        model_choice = st.selectbox("Base Pretrained Architecture", ["ResNet50", "VGG16", "MobileNetV2"])
