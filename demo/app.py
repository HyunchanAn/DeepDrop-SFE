import streamlit as st
import cv2
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ai_engine import AIContactAngleAnalyzer
from physics_engine import DropletPhysics
from perspective import PerspectiveCorrector

# Page Config
st.set_page_config(page_title="DeepDrop-AnyView", layout="wide")

st.title("DeepDrop-AnyView: 임의 각도 표면 에너지 분석기")
st.markdown("""
> **안내**: 이 시스템은 **기준 물체(예: 100원 동전)**를 사용하여 사진의 원근 왜곡을 보정합니다.
> 액적 옆에 동전을 두고, 동전이 잘 보이도록 촬영해 주세요.
""")

# Sidebar
st.sidebar.header("설정 (Configuration)")

# Experiment Parameters
st.sidebar.subheader("실험 변수")
volume_ul = st.sidebar.number_input("액적 부피 (Droplet Volume, µL)", min_value=0.1, value=3.0, step=0.1)

# Reference Object
st.sidebar.subheader("기준 물체 (Reference Object)")
ref_options = {
    "100원 동전 (Old)": 24.0, # 100 KRW
    "100원 동전 (New)": 24.0,
    "500원 동전": 26.5,
    "사용자 지정 (Custom)": 0.0
}
ref_choice = st.sidebar.selectbox("기준 물체 선택", list(ref_options.keys()))

if ref_choice == "사용자 지정 (Custom)":
    real_diameter_mm = st.sidebar.number_input("물체 직경 (Diameter, mm)", min_value=1.0, value=10.0)
else:
    real_diameter_mm = ref_options[ref_choice]
    st.sidebar.info(f"직경: {real_diameter_mm} mm")

# Liquid Type
liquid_type = st.sidebar.selectbox("액체 종류 (Liquid Type)", list(DropletPhysics.LIQUID_DATA.keys()))

# Model Loading
@st.cache_resource
def load_models():
    # Helper to download model if not exists
    model_path = os.path.join(os.path.dirname(__file__), "../models/mobile_sam.pt")
    if not os.path.exists(model_path):
        import requests
        st.info("MobileSAM 모델 다운로드 중...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
        r = requests.get(url)
        with open(model_path, 'wb') as f:
            f.write(r.content)
        st.success("모델 다운로드 완료!")
        
    try:
        analyzer = AIContactAngleAnalyzer(model_path)
        corrector = PerspectiveCorrector()
        return analyzer, corrector
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None, None

analyzer, corrector = load_models()

if not analyzer:
    st.stop()

# Main Workflow
uploaded_file = st.file_uploader("이미지 업로드 (동전 & 액적 포함)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 1. Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    st.subheader("1. 기준 물체 감지 (Reference Detection)")
    
    # Auto-detect coin
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_rgb, caption="원본 이미지", use_container_width=True)
        
    with st.spinner("이미지에서 동전을 찾는 중입니다..."):
        coin_box = analyzer.auto_detect_coin_candidate(image)
        
    if coin_box is not None:
        # Draw box for visualization
        preview_img = image_rgb.copy()
        x1, y1, x2, y2 = coin_box
        cv2.rectangle(preview_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        with col2:
            st.image(preview_img, caption="감지된 기준 물체", use_container_width=True)
            
        st.info("녹색 박스가 동전을 정확히 감지했나요?")
        if st.checkbox("기준 물체 확인 (Confirm)", value=True):
            
            # 2. Perspective Correction
            st.subheader("2. 원근 보정 및 변환 (Homography)")
            
            # Generate detailed mask for homography
            analyzer.set_image(image_rgb)
            coin_mask, _ = analyzer.predict_mask(box=coin_box)
            coin_mask_binary = analyzer.get_binary_mask(coin_mask)
            
            # Calculate Homography
            H, warped_size, coin_info = corrector.find_homography(image_rgb, coin_mask_binary)
            
            if H is not None:
                warped_img = corrector.warp_image(image_rgb, H, warped_size)
                
                # Visualize Warped Image
                col_w1, col_w2 = st.columns(2)
                with col_w1:
                    st.image(warped_img, caption="보정된 Top-View 이미지", use_container_width=True)
                    
                # 3. Droplet Analysis
                with col_w2:
                    st.write("보정된 이미지에서 액적 분석 중...")
                    
                    # Analyze Droplet on Warped Image
                    analyzer.set_image(warped_img)
                    
                    # Assume droplet is near center or just use center point
                    droplet_mask, drop_score = analyzer.predict_mask()
                    
                    # Visualization
                    vis_mask = np.zeros_like(warped_img)
                    vis_mask[droplet_mask] = [255, 0, 0] # Red mask
                    overlay = cv2.addWeighted(warped_img, 0.7, vis_mask, 0.3, 0)
                    st.image(overlay, caption="액적 세그멘테이션 결과", use_container_width=True)
                
                # 4. Calculation
                st.subheader("3. 측정 결과 (Analysis Report)")
                
                # Get scale
                (cx, cy, radius_px) = coin_info
                pixels_per_mm = DropletPhysics.calculate_pixels_per_mm(radius_px, real_diameter_mm)
                
                # Get Contact Diameter
                contact_diameter_mm = DropletPhysics.calculate_contact_diameter(droplet_mask, pixels_per_mm)
                
                # Get Contact Angle
                contact_angle = DropletPhysics.calculate_contact_angle(volume_ul, contact_diameter_mm)
                
                # Display Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Pixel Scale", f"{pixels_per_mm:.1f} px/mm")
                m2.metric("접촉 직경 (Diameter)", f"{contact_diameter_mm:.2f} mm")
                m3.metric("접촉각 (Contact Angle)", f"{contact_angle:.1f}°")
                
                st.success(f"분석 완료: **{contact_angle:.1f}°**")
                
            else:
                st.error("원근 보정 실패. 동전이 찌그러져 있거나 윤곽선이 불분명합니다.")

    else:
        st.error("동전을 찾을 수 없습니다. 조명이 밝고 동전이 선명한 사진을 사용해 주세요.")
        st.image(image_rgb, caption="입력 이미지", use_container_width=True)
