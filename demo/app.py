import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import sys

# 프로젝트 루트 경로 추가 (모듈 import 위함)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ai_engine import AICntactAngleAnalyzer
from src.physics_engine import DropletPhysics

# --- Page Config ---
st.set_page_config(
    page_title="DeepDrop Analyzer",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4A90E2; 
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://via.placeholder.com/300x100?text=DeepDrop+Logo", use_column_width=True)
    st.title("Settings")
    
    model_type = st.selectbox("AI Model Type", ["vit_t (MobileSAM)", "vit_h (Heavy)"])
    st.info(f"Loaded: {model_type}")
    
    st.divider()
    
    st.subheader("Experimental Setup")
    liquid_type = st.radio("Liquid Type", ["Water", "Diiodomethane", "Other"])
    
    st.divider()
    st.caption("v1.0.0 | Powered by MobileSAM")

# --- Initialize Engines (Singleton-ish with st.cache_resource) ---
@st.cache_resource
def load_ai_model():
    # 모델 경로 확인
    model_path = os.path.join("models", "mobile_sam.pt")
    if not os.path.exists(model_path):
        return None
    try:
        analyzer = AICntactAngleAnalyzer(model_path, "vit_t")
        return analyzer
    except Exception as e:
        return str(e)

analyzer = load_ai_model()

# --- Main Content ---
st.markdown('<div class="main-header">DeepDrop Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Based Surface Free Energy Analysis System</div>', unsafe_allow_html=True)

if analyzer is None:
    st.error("⚠️ MobileSAM Model not found. Please place `mobile_sam.pt` in the `models/` directory.")
    st.stop()
elif isinstance(analyzer, str): # Error message
    st.error(f"⚠️ Error loading model: {analyzer}")
    st.stop()

# Layout
col1, col2 = st.columns([1, 1])

# State management for OWRK
if 'water_angle' not in st.session_state:
    st.session_state.water_angle = None
if 'diiodo_angle' not in st.session_state:
    st.session_state.diiodo_angle = None

with col1:
    st.subheader("1. Setup & Upload")
    uploaded_file = st.file_uploader("Upload Droplet Image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        # Load Image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        st.image(image_rgb, caption="Original Image", use_column_width=True)
        
        if st.button("🚀 Analyze Droplet", type="primary"):
            with st.spinner("AI Segmenting & Profiling..."):
                # 1. AI Segmentation
                analyzer.set_image(image_rgb)
                mask, score = analyzer.predict_mask() # Center point prompt
                
                # 2. Physics Profiling
                binary_mask = analyzer.get_binary_mask(mask)
                points = DropletPhysics.extract_boundary_points(binary_mask)
                
                if points is not None:
                    ellipse = DropletPhysics.fit_ellipse(points)
                    contact_angle = DropletPhysics.calculate_contact_angle(ellipse, 0)
                    
                    # Visualization
                    res_img = image_rgb.copy()
                    
                    # Draw Contour
                    cv2.drawContours(res_img, [points], -1, (0, 255, 0), 2)
                    
                    # Draw Ellipse
                    if ellipse:
                        cv2.ellipse(res_img, ellipse, (255, 0, 0), 2)
                    
                    # Save results to session state
                    if liquid_type == "Water":
                        st.session_state.water_angle = contact_angle
                    elif liquid_type == "Diiodomethane":
                        st.session_state.diiodo_angle = contact_angle
                    
                    st.success("Analysis Complete!")
                    
                    with col2:
                        st.subheader("2. Analysis Results")
                        st.image(res_img, caption=f"AI Segmentation (Confidence: {score:.2f})", use_column_width=True)
                        
                        # Metrics
                        m1, m2 = st.columns(2)
                        with m1:
                            st.metric("Contact Angle", f"{contact_angle:.2f}°")
                        with m2:
                            st.metric("AI Confidence", f"{score:.1%}")
                            
                        # OWRK Calculation Trigger
                        st.divider()
                        st.markdown("### Total Surface Energy (OWRK)")
                        
                        if st.session_state.water_angle and st.session_state.diiodo_angle:
                            total_sfe, sd, sp = DropletPhysics.calculate_owrk(
                                st.session_state.water_angle, 
                                st.session_state.diiodo_angle
                            )
                            st.metric("Total SFE", f"{total_sfe:.2f} mN/m")
                            st.progress(min(total_sfe/100, 1.0))
                            st.write(f"- Dispersive: {sd:.2f} mN/m")
                            st.write(f"- Polar: {sp:.2f} mN/m")
                        else:
                            st.warning("Needs both Water and Diiodomethane data to calculate SFE.")
                            st.write(f"Water: {st.session_state.water_angle if st.session_state.water_angle else 'Not Measured'}")
                            st.write(f"Diiodo: {st.session_state.diiodo_angle if st.session_state.diiodo_angle else 'Not Measured'}")
