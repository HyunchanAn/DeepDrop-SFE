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

# --- Language Resource ---
TRANS = {
    "KR": {
        "title": "DeepDrop-AnyView: 임의 각도 표면 에너지 분석기",
        "notice": """
        > **안내**: 이 시스템은 **기준 물체(예: 100원 동전)**를 사용하여 사진의 원근 왜곡을 보정합니다.
        > 액적 옆에 동전을 두고, 동전이 잘 보이도록 촬영해 주세요.
        """,
        "header_config": "설정 (Configuration)",
        "header_exp_params": "실험 변수",
        "lbl_volume": "액적 부피 (Droplet Volume, µL)",
        "header_ref_obj": "기준 물체 (Reference Object)",
        "lbl_ref_choice": "기준 물체 선택",
        "opt_100_old": "100원 동전 (Old)",
        "opt_100_new": "100원 동전 (New)",
        "opt_500": "500원 동전",
        "opt_custom": "사용자 지정 (Custom)",
        "lbl_ref_diam": "물체 직경 (Diameter, mm)",
        "msg_diam": "직경: {} mm",
        "lbl_liquid": "액체 종류 (Liquid Type)",
        "msg_downloading": "MobileSAM 모델 다운로드 중...",
        "msg_download_done": "모델 다운로드 완료!",
        "err_model_load": "모델 로드 실패: {}",
        "lbl_upload": "이미지 업로드 (동전 & 액적 포함)",
        "header_step1": "1. 기준 물체 감지 (Reference Detection)",
        "cap_original": "원본 이미지",
        "msg_detecting": "이미지에서 동전을 찾는 중입니다...",
        "cap_detected": "감지된 기준 물체",
        "msg_confirm_box": "녹색 박스가 동전을 정확히 감지했나요?",
        "chk_confirm": "기준 물체 확인 (Confirm)",
        "header_step2": "2. 원근 보정 및 변환 (Homography)",
        "cap_warped": "보정된 Top-View 이미지",
        "msg_analyzing": "보정된 이미지에서 액적 분석 중...",
        "cap_segmentation": "액적 세그멘테이션 결과",
        "header_step3": "3. 측정 결과 (Analysis Report)",
        "lbl_pixel_scale": "픽셀 스케일",
        "lbl_diameter": "접촉 직경",
        "lbl_angle": "접촉각",
        "msg_success": "분석 완료: **{:.1f}°**",
        "err_homography": "원근 보정 실패. 동전이 찌그러져 있거나 윤곽선이 불분명합니다.",
        "err_no_coin": "동전을 찾을 수 없습니다. 조명이 밝고 동전이 선명한 사진을 사용해 주세요.",
        "cap_input": "입력 이미지"
    },
    "EN": {
        "title": "DeepDrop-AnyView: Arbitrary Angle SFE Analyzer",
        "notice": """
        > **Note**: This system uses a **Reference Object (e.g., Coin)** to correct perspective distortion.
        > Please place the coin next to the droplet and ensure it is clearly visible.
        """,
        "header_config": "Configuration",
        "header_exp_params": "Experimental Parameters",
        "lbl_volume": "Droplet Volume (µL)",
        "header_ref_obj": "Reference Object",
        "lbl_ref_choice": "Select Reference Object",
        "opt_100_old": "100 KRW Coin (Old)",
        "opt_100_new": "100 KRW Coin (New)",
        "opt_500": "500 KRW Coin",
        "opt_custom": "Custom Size",
        "lbl_ref_diam": "Diameter (mm)",
        "msg_diam": "Diameter: {} mm",
        "lbl_liquid": "Liquid Type",
        "msg_downloading": "Downloading MobileSAM model...",
        "msg_download_done": "Model downloaded!",
        "err_model_load": "Failed to load models: {}",
        "lbl_upload": "Upload Image (with Coin & Droplet)",
        "header_step1": "1. Reference Object Detection",
        "cap_original": "Original Image",
        "msg_detecting": "Detecting reference object...",
        "cap_detected": "Detected Reference Candidate",
        "msg_confirm_box": "Is the green box correctly highlighting the object?",
        "chk_confirm": "Confirm Reference Object",
        "header_step2": "2. Perspective Correction",
        "cap_warped": "Warped Image (Top-View)",
        "msg_analyzing": "Analyzing Droplet on Warped Image...",
        "cap_segmentation": "Droplet Segmentation",
        "header_step3": "3. Measurement Results",
        "lbl_pixel_scale": "Pixel Scale",
        "lbl_diameter": "Contact Diameter",
        "lbl_angle": "Contact Angle",
        "msg_success": "Analysis Complete: **{:.1f}°**",
        "err_homography": "Homography failed. The coin might be unclear or not circular.",
        "err_no_coin": "Could not auto-detect reference object. Ensure good lighting.",
        "cap_input": "Input Image"
    }
}

# Language Selection
lang_code = st.sidebar.radio("Language / 언어", ["KR", "EN"], horizontal=True)
R = TRANS[lang_code]

st.title(R["title"])
st.markdown(R["notice"])

# Sidebar
st.sidebar.header(R["header_config"])

# Experiment Parameters
st.sidebar.subheader(R["header_exp_params"])
volume_ul = st.sidebar.number_input(R["lbl_volume"], min_value=0.1, value=3.0, step=0.1)

# Reference Object
st.sidebar.subheader(R["header_ref_obj"])
ref_options = {
    R["opt_100_old"]: 24.0, # 100 KRW
    R["opt_100_new"]: 24.0,
    R["opt_500"]: 26.5,
    R["opt_custom"]: 0.0
}
ref_choice = st.sidebar.selectbox(R["lbl_ref_choice"], list(ref_options.keys()))

if ref_choice == R["opt_custom"]:
    real_diameter_mm = st.sidebar.number_input(R["lbl_ref_diam"], min_value=1.0, value=10.0)
else:
    real_diameter_mm = ref_options[ref_choice]
    st.sidebar.info(R["msg_diam"].format(real_diameter_mm))

# Liquid Type
liquid_type = st.sidebar.selectbox(R["lbl_liquid"], list(DropletPhysics.LIQUID_DATA.keys()))

# Model Loading
@st.cache_resource
def load_models():
    # Helper to download model if not exists
    model_path = os.path.join(os.path.dirname(__file__), "../models/mobile_sam.pt")
    if not os.path.exists(model_path):
        import requests
        st.info(R["msg_downloading"])
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
        r = requests.get(url)
        with open(model_path, 'wb') as f:
            f.write(r.content)
        st.success(R["msg_download_done"])
        
    try:
        analyzer = AIContactAngleAnalyzer(model_path)
        corrector = PerspectiveCorrector()
        return analyzer, corrector
    except Exception as e:
        st.error(R["err_model_load"].format(e))
        return None, None

analyzer, corrector = load_models()

if not analyzer:
    st.stop()

# Main Workflow
uploaded_file = st.file_uploader(R["lbl_upload"], type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 1. Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    st.subheader(R["header_step1"])
    
    # Auto-detect coin
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_rgb, caption=R["cap_original"], use_container_width=True)
        
    with st.spinner(R["msg_detecting"]):
        coin_box = analyzer.auto_detect_coin_candidate(image)
        
    if coin_box is not None:
        # Draw box for visualization
        preview_img = image_rgb.copy()
        x1, y1, x2, y2 = coin_box
        cv2.rectangle(preview_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        with col2:
            st.image(preview_img, caption=R["cap_detected"], use_container_width=True)
            
        st.info(R["msg_confirm_box"])
        if st.checkbox(R["chk_confirm"], value=True):
            
            # 2. Perspective Correction
            st.subheader(R["header_step2"])
            
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
                    st.image(warped_img, caption=R["cap_warped"], use_container_width=True)
                    
                # 3. Droplet Analysis
                with col_w2:
                    st.write(R["msg_analyzing"])
                    
                    # Analyze Droplet on Warped Image
                    analyzer.set_image(warped_img)
                    
                    # Assume droplet is near center or just use center point
                    droplet_mask, drop_score = analyzer.predict_mask()
                    
                    # Visualization
                    vis_mask = np.zeros_like(warped_img)
                    vis_mask[droplet_mask] = [255, 0, 0] # Red mask
                    overlay = cv2.addWeighted(warped_img, 0.7, vis_mask, 0.3, 0)
                    st.image(overlay, caption=R["cap_segmentation"], use_container_width=True)
                
                # 4. Calculation
                st.subheader(R["header_step3"])
                
                # Get scale
                (cx, cy, radius_px) = coin_info
                pixels_per_mm = DropletPhysics.calculate_pixels_per_mm(radius_px, real_diameter_mm)
                
                # Get Contact Diameter
                contact_diameter_mm = DropletPhysics.calculate_contact_diameter(droplet_mask, pixels_per_mm)
                
                # Get Contact Angle
                contact_angle = DropletPhysics.calculate_contact_angle(volume_ul, contact_diameter_mm)
                
                # Display Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric(R["lbl_pixel_scale"], f"{pixels_per_mm:.1f} px/mm")
                m2.metric(R["lbl_diameter"], f"{contact_diameter_mm:.2f} mm")
                m3.metric(R["lbl_angle"], f"{contact_angle:.1f}°")
                
                st.success(R["msg_success"].format(contact_angle))
                
            else:
                st.error(R["err_homography"])

    else:
        st.error(R["err_no_coin"])
        st.image(image_rgb, caption=R["cap_input"], use_container_width=True)
