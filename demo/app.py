import streamlit as st
import cv2
import numpy as np
import sys
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

from deepdrop_sfe import AIContactAngleAnalyzer, DropletPhysics, PerspectiveCorrector


def import_image_pil(img_rgb):
    return Image.fromarray(img_rgb)

# --- Language Resource ---
TRANS = {
    "KR": {
        "title": "DeepDrop-AnyView: 임의 각도 표면 에너지 분석기",
        "notice": """
        > **안내**: 이 시스템은 기준 물체(예: 100원 동전)를 사용하여 사진의 원근 왜곡을 보정합니다.
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
        "msg_downloading": "MobileSAM 모델 로딩 중...",
        "msg_download_done": "모델 로드 완료!",
        "err_model_load": "MobileSAM 모델 로드 실패: {}",
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
        "msg_success": "분석 완료: {:.1f}°",
        "header_sfe_table": "데이터 관리 및 SFE (Data & SFE)",
        "btn_add": "결과 추가 (Add to Table)",
        "btn_clear": "초기화 (Clear)",
        "lbl_total_sfe": "총 표면 에너지",
        "lbl_disperse": "분산 성분",
        "lbl_polar": "극성 성분",
        "btn_export_csv": "CSV로 내보내기",
        "btn_export_json": "JSON으로 내보내기",
        "err_homography": "원근 보정 실패. 동전이 찌그러져 있거나 윤곽선이 불분명합니다.",
        "err_no_coin": "동전을 찾을 수 없습니다. 조명이 밝고 동전이 선명한 사진을 사용해 주세요.",
        "cap_input": "입력 이미지",
        "header_drop_detect": "액적 위치 지정 (Droplet Localization)",
        "lbl_drop_mode": "액적 찾는 방법",
        "opt_drop_auto": "자동 감지 (Advanced Auto)",
        "opt_drop_manual": "수동 선택 (Box Draw)",
        "msg_drop_confirm": "액적이 빨간색으로 잘 표시되었나요? 아니면 수동으로 선택해 주세요.",
        "lbl_advanced_diag": "고급 진단 정보 (Advanced Diagnostics)",
        "lbl_circularity": "원형도 (Circularity)",
        "msg_low_reliability": "주의: 원형도 점수가 낮습니다. 표면이 불균일하거나 원근 보정이 정확하지 않을 수 있습니다."
    },
    "EN": {
        "title": "DeepDrop-AnyView: Arbitrary Angle SFE Analyzer",
        "notice": """
        > Note: This system uses a Reference Object (e.g., Coin) to correct perspective distortion.
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
        "msg_downloading": "Loading MobileSAM...",
        "msg_download_done": "Model loaded!",
        "err_model_load": "Failed to load MobileSAM: {}",
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
        "msg_success": "Analysis Complete: {:.1f}°",
        "header_sfe_table": "Data Management & SFE",
        "btn_add": "Add to Table",
        "btn_clear": "Clear Table",
        "lbl_total_sfe": "Total SFE",
        "lbl_disperse": "Dispersive Component",
        "lbl_polar": "Polar Component",
        "btn_export_csv": "Export to CSV",
        "btn_export_json": "Export to JSON",
        "err_homography": "Homography failed. The coin might be unclear or not circular.",
        "err_no_coin": "Could not auto-detect reference object. Ensure good lighting.",
        "cap_input": "Input Image",
        "header_drop_detect": "Droplet Localization",
        "lbl_drop_mode": "Droplet Detection Mode",
        "opt_drop_auto": "Advanced Auto Detection",
        "opt_drop_manual": "Manual Selection (Box Draw)",
        "msg_drop_confirm": "Is the droplet correctly highlighted in red? If not, use manual mode.",
        "lbl_advanced_diag": "Advanced Diagnostics",
        "lbl_circularity": "Circularity Score",
        "msg_low_reliability": "Caution: Low circularity score. Surface roughness or perspective error may exist."
    }
}

# Language Selection
from streamlit_javascript import st_javascript

# ... (Previous imports)

# Language Selection
# Move language selection to top level so it is always accessible or keep in sidebar/expander?
# Let's keep it in the rendering function or separate.
# Actually, Lang selection is global. Let's put it in the control function too or just keep it in sidebar for PC and main for Mobile.

def render_controls(container, R):
    container.header(R["header_config"])
    
    # Language (Optional: move here if we want it in the same block)
    # lang_code = container.radio("Language / 언어", ["KR", "EN"], horizontal=True, key="lang_select")
    # But R depends on lang_code, so lang_code must be decided BEFORE calling this or passed in?
    # R is global currently. Let's just render the config interactions.

    # Experiment Parameters
    container.subheader(R["header_exp_params"])
    volume_ul = container.number_input(R["lbl_volume"], min_value=0.1, max_value=1000.0, value=100.0, step=0.1, format="%.1f", key="ctrl_volume")
    
    # Reference Object
    container.subheader(R["header_ref_obj"])
    ref_options = {
        R["opt_100_old"]: 24.0, # 100 KRW
        R["opt_100_new"]: 24.0,
        R["opt_500"]: 26.5,
        "10원 동전 (Small)": 18.0,
        "10원 동전 (Large)": 22.86,
        R["opt_custom"]: 0.0
    }
    ref_choice = container.selectbox(R["lbl_ref_choice"], list(ref_options.keys()), key="ctrl_ref_choice")
    
    real_diameter_mm = 0.0
    if ref_choice == R["opt_custom"]:
        real_diameter_mm = container.number_input(R["lbl_ref_diam"], min_value=1.0, value=10.0, key="ctrl_custom_diam")
    else:
        real_diameter_mm = ref_options[ref_choice]
        container.info(R["msg_diam"].format(real_diameter_mm))
    
    # Liquid Type
    liquid_type = container.selectbox(R["lbl_liquid"], list(DropletPhysics.LIQUID_DATA.keys()), key="ctrl_liquid")
    
    return volume_ul, real_diameter_mm, liquid_type

# Screen Width Detection
# 0 means return value explicitly (optional default)
ui_width = st_javascript("window.innerWidth")

# Setup Language First (Global)
# We can just put this in the sidebar always? Or if mobile, maybe top of page?
# User wants "Left side functions to Main screen" on mobile.
# Let's handle language selection before layout decision to get `R`.

if ui_width is not None and ui_width < 800:
    # Mobile Mode
    is_mobile = True
else:
    # PC Mode (Default)
    is_mobile = False

# Helper: Draw Dynamic Droplet Profile
def plot_droplet_profile(contact_angle):
    fig, ax = plt.subplots(figsize=(4, 2))
    theta = np.radians(contact_angle)
    
    # Baseline
    ax.plot([-1.5, 1.5], [0, 0], color='black', linewidth=2)
    
    base_radius = 1.0
    
    if contact_angle < 90:
        # Hydrophilic
        R = base_radius / np.sin(theta)
        y_c = -base_radius / np.tan(theta)
        
        # Arc top
        circle = plt.Circle((0, y_c), R, color='skyblue', alpha=0.6)
        ax.add_patch(circle)
        ax.set_ylim(0, 2.0)
        
    else:
        # Hydrophobic
        R = base_radius / np.sin(np.pi - theta)
        y_c = -base_radius / np.tan(theta) 
        
        circle = plt.Circle((0, y_c), R, color='skyblue', alpha=0.6)
        ax.add_patch(circle)
        ax.set_ylim(0, 2.5)

    ax.set_xlim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Visualized Profile (Angle: {contact_angle:.1f}°)", fontsize=10)
    plt.tight_layout()
    return fig

# Render Language Toggle
# For mobile, maybe small columns at top?
if is_mobile:
    lang_code = st.radio("Language / 언어", ["KR", "EN"], horizontal=True, key="lang_main")
else:
    lang_code = st.sidebar.radio("Language / 언어", ["KR", "EN"], horizontal=True, key="lang_side")
    
R = TRANS[lang_code]

st.title(R["title"])
st.markdown(R["notice"])

# Render Controls
if is_mobile:
    with st.expander("⚙️ " + R["header_config"] + " (Settings)", expanded=False):
        volume_ul, real_diameter_mm, liquid_type = render_controls(st, R)
else:
    volume_ul, real_diameter_mm, liquid_type = render_controls(st.sidebar, R)

# Model Loading
@st.cache_resource
def load_models():
    """
    MobileSAM 모델 로드 (ai_engine.py 연동)
    """
    try:
        with st.spinner(R["msg_downloading"]):
            analyzer = AIContactAngleAnalyzer(model_type="vit_t")
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
    
    # --- Manual Override UI ---
    if 'ref_cx' not in st.session_state: st.session_state.ref_cx = int(image_rgb.shape[1] // 2)
    if 'ref_cy' not in st.session_state: st.session_state.ref_cy = int(image_rgb.shape[0] // 2)
    if 'ref_r' not in st.session_state: st.session_state.ref_r = int(image_rgb.shape[0] // 5)
    
    col_auto, col_manual = st.columns([1, 2])
    
    with col_auto:
        if st.button("🔄 자동 감지 실행 (Auto Detect)", use_container_width=True):
             with st.spinner(R["msg_detecting"]):
                coin_box, circle_info = analyzer.auto_detect_coin_candidate(image)
                if circle_info:
                    st.session_state.ref_cx = int(circle_info[0])
                    st.session_state.ref_cy = int(circle_info[1])
                    st.session_state.ref_r = int(circle_info[2])
                    st.success("동전 감지 성공!")
                else:
                    st.error("자동 감지 실패. 수동으로 조절해주세요.")

    h, w, _ = image_rgb.shape
    
    with col_manual:
        st.info("슬라이더로 초록색 원을 동전에 맞춰주세요. (Adjust green circle to coin)")
        
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        st.session_state.ref_cx = st.slider("Center X", 0, w, st.session_state.ref_cx, key="slider_cx")
    with s_col2:
        st.session_state.ref_cy = st.slider("Center Y", 0, h, st.session_state.ref_cy, key="slider_cy")
    with s_col3:
        st.session_state.ref_r = st.slider("Radius", 10, w//2, st.session_state.ref_r, key="slider_r")

    # Preview Overlay
    preview_img = image_rgb.copy()
    cv2.circle(preview_img, (st.session_state.ref_cx, st.session_state.ref_cy), st.session_state.ref_r, (0, 255, 0), 2)
    cv2.circle(preview_img, (st.session_state.ref_cx, st.session_state.ref_cy), 5, (0, 0, 255), -1)
    
    st.image(preview_img, caption="Reference Object Alignment", use_container_width=True)
    
    # Confirm
    coin_box = None
    if st.checkbox("위치 확정 및 분석 시작 (Confirm & Analyze)", value=False):
        # Create box from circle for compatibility
        cx, cy, r = st.session_state.ref_cx, st.session_state.ref_cy, st.session_state.ref_r
        coin_box = np.array([cx - r, cy - r, cx + r, cy + r])
        
        # Manually create binary mask (100% Geometry Trust)
        manual_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(manual_mask, (cx, cy), r, 1, -1)
        manual_mask_bool = manual_mask.astype(bool) 


    if coin_box is not None:
        # 2. Perspective Correction
        st.subheader(R["header_step2"])
            
        # If manual mask is ready, use it. Otherwise predict (should not happen with new flow)
        if 'manual_mask_bool' in locals():
            coin_mask = manual_mask_bool
        else:
             analyzer.set_image(image_rgb)
             coin_mask, _ = analyzer.predict_mask(box=coin_box)
        
        coin_mask_binary = analyzer.get_binary_mask(coin_mask)
        
        # --- Advanced Validation on Mask ---
        contours, _ = cv2.findContours(coin_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            if len(cnt) >= 5:
                (ex, ey), (eda, edb), e_angle = cv2.fitEllipse(cnt)
                e_ar = max(eda, edb) / (min(eda, edb) + 1e-6)
            else:
                e_ar = 100.0
            
            if solidity < 0.8:
                st.warning(f"⚠️ 감지된 물체의 모양이 불규칙합니다 (Solidity: {solidity:.2f}). 조명이 반사되거나 배경이 복잡할 수 있습니다.")
            if e_ar > 5.0:
                st.warning(f"⚠️ 물체가 너무 납작해 보입니다 (AR: {e_ar:.1f}). 카메라 각도가 너무 기울어졌을 수 있습니다.")
                
        # DEBUG: Visualize Coin Mask
        if np.sum(coin_mask_binary) > 0:
             # st.image(coin_mask_binary * 255, caption="Debug: Coin Mask (Binary)", use_container_width=True)
             pass
        else:
             st.warning("동전 마스크 추출 실패 (Empty Mask). 원인: 면적/원형도 필터 탈락")

        # Calculate Homography
        H, warped_size, coin_info, fitted_ellipse = corrector.find_homography(image_rgb, coin_mask_binary)
        
        if H is not None:
            # DEBUG: Visualize Ellipse Fit on Original Image
            debug_ellipse_img = image_rgb.copy()
            (ecx, ecy), (eda, edb), eangle = fitted_ellipse
            # Draw ellipse
            cv2.ellipse(debug_ellipse_img, ((ecx, ecy), (eda, edb), eangle), (255, 0, 0), 2)
            # Draw center
            cv2.circle(debug_ellipse_img, (int(ecx), int(ecy)), 5, (0, 0, 255), -1)
            
            st.image(debug_ellipse_img, caption="Debug: Fitted Ellipse", use_container_width=True)

            warped_img = corrector.warp_image(image_rgb, H, warped_size)
            
            # Visualize Warped Image
            st.image(warped_img, caption=R["cap_warped"], use_container_width=True)
            
            # 3. Droplet Analysis
            st.divider()
            st.subheader(R["header_drop_detect"])
            
            drop_mode = st.radio(R["lbl_drop_mode"], [R["opt_drop_auto"], R["opt_drop_manual"]], horizontal=True)
            droplet_box = None
            
            if drop_mode == R["opt_drop_auto"]:
                with st.spinner(R["msg_detecting"]):
                    # Use new auto detection, excluding coin area
                    # coin_box might be in original image coords. Warp it? 
                    # Easier to just let auto_detect find objects in warped image.
                    droplet_box = analyzer.auto_detect_droplet_candidate(warped_img)
            
            elif drop_mode == R["opt_drop_manual"]:
                from streamlit_drawable_canvas import st_canvas
                h_w, w_w, _ = warped_img.shape
                d_w = 450
                s_w = d_w / w_w
                d_h = int(h_w * s_w)
                
                # Ensure res_w is explicitly uint8 for PIL and matches the format
                res_w = cv2.resize(warped_img, (d_w, d_h)).astype(np.uint8)
                
                c_col1, c_col2 = st.columns(2)
                with c_col1:
                    canvas_drop = st_canvas(
                        fill_color="rgba(255, 0, 0, 0.3)",
                        stroke_width=2,
                        stroke_color="#FF0000",
                        background_image=import_image_pil(res_w),
                        update_streamlit=True,
                        height=d_h,
                        width=d_w,
                        drawing_mode="rect",
                        key=f"canvas_drop_{uploaded_file.name}"
                    )
                
                if canvas_drop.json_data is not None:
                    objs = pd.json_normalize(canvas_drop.json_data["objects"])
                    if not objs.empty:
                        o = objs.iloc[-1]
                        droplet_box = np.array([
                            int(o["left"] / s_w),
                            int(o["top"] / s_w),
                            int((o["left"] + o["width"]) / s_w),
                            int((o["top"] + o["height"]) / s_w)
                        ])
            
            # Analyze Droplet on Warped Image
            analyzer.set_image(warped_img)
            
            with st.spinner(R["msg_analyzing"]):
                droplet_mask, drop_score = analyzer.predict_mask(box=droplet_box)
            
            # Visualization
            vis_mask = np.zeros_like(warped_img)
            droplet_mask_bool = droplet_mask.astype(bool)  # Convert to boolean for indexing
            vis_mask[droplet_mask_bool] = [255, 0, 0] # Red mask
            overlay = cv2.addWeighted(warped_img, 0.7, vis_mask, 0.3, 0)
            
            if drop_mode == R["opt_drop_manual"]:
                with c_col2:
                    st.image(overlay, caption=R["cap_segmentation"], use_container_width=True)
            else:
                st.image(overlay, caption=R["cap_segmentation"], use_container_width=True)
            
            # 4. Calculation
            st.subheader(R["header_step3"])
            
            # Get scale
            (cx, cy, radius_px) = coin_info
            pixels_per_mm = DropletPhysics.calculate_pixels_per_mm(radius_px, real_diameter_mm)
            
            # Get Contact Diameter and Circularity
            contact_diameter_mm, circularity = DropletPhysics.calculate_contact_diameter(droplet_mask, pixels_per_mm, return_extra=True)
            
            # Get Contact Angle with Info
            contact_angle, diag = DropletPhysics.calculate_contact_angle(volume_ul, contact_diameter_mm, return_info=True)
            
            # Display Metrics
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric(R["lbl_pixel_scale"], f"{pixels_per_mm:.1f} px/mm")
            m_col2.metric(R["lbl_diameter"], f"{contact_diameter_mm:.2f} mm")
            m_col3.metric(R["lbl_circularity"], f"{circularity:.3f}")
            m_col4.metric(R["lbl_angle"], f"{contact_angle:.1f}°")
            
            if circularity < 0.9:
                st.warning(R["msg_low_reliability"])
            
            st.success(R["msg_success"].format(contact_angle))

            # Show Dynamic Plot
            st.pyplot(plot_droplet_profile(contact_angle))
            
            # Advanced Diagnostics
            with st.expander(R["lbl_advanced_diag"]):
                st.write(f"Solver Status: {diag['status']}")
                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    st.write(f"- Droplet Radius (r): `{diag['r']:.4f}` mm")
                    st.write(f"- Target Volume (V): `{diag['target_V']:.4f}` µL")
                    st.write(f"- Full Sphere Vol: `{diag['v_full']:.4f}` µL")
                with d_col2:
                    st.write(f"- v_low (at 0°): `{diag['v_low']:.4f}`")
                    st.write(f"- v_high (at 180°): `{diag['v_high']:.4e}`")
                
                if diag['status'] != "Success":
                    st.warning("계산에 실패하거나 캡(Cap)이 씌워졌습니다. 위 값들을 개발자에게 전달해 주세요.")
            
            # 5. Reference Guide
            st.divider()
            with st.expander("ℹ️ 접촉각 참조 가이드 (Contact Angle Reference)", expanded=True):
                st.markdown("""
                ### 접촉각의 의미 (Wetting Properties)
                * **0° ~ 10°**: 완전 퍼짐 (Super-hydrophilic) - 물이 표면에 쫙 달라붙음.
                * **10° ~ 90°**: 친수성 (Hydrophilic) - 물이 어느 정도 퍼짐.
                * **90° ~ 150°**: 소수성 (Hydrophobic) - 물방울이 맺힘.
                * **150° ~ 180°**: 초소수성 (Super-hydrophobic) - 물방울이 구슬처럼 굴러다님.
                """)
                # Local Image (Generated by Matplotlib)
                st.image("demo/assets/contact_angle_ref.png", 
                         caption="접촉각(θ)과 젖음성(Wetting) 예시", 
                         use_container_width=True)
            st.divider()
            st.subheader(R["header_sfe_table"])
            
            if 'measurements' not in st.session_state:
                st.session_state.measurements = []
            
            c1, c2 = st.columns(2)
            with c1:
                if st.button(R["btn_add"]):
                    st.session_state.measurements.append({
                        "Liquid": liquid_type,
                        "Angle": round(contact_angle, 2),
                        "Volume": volume_ul,
                        "Diameter": round(contact_diameter_mm, 2)
                    })
            with c2:
                if st.button(R["btn_clear"]):
                    st.session_state.measurements = []
            
            if st.session_state.measurements:
                df = pd.DataFrame(st.session_state.measurements)
                st.table(df)
                
                # Calculate SFE if at least 2 liquids
                if len(df['Liquid'].unique()) >= 2:
                    m_list = [{"liquid": row["Liquid"], "angle": row["Angle"]} for _, row in df.iterrows()]
                    total_sfe, gamma_d, gamma_p = DropletPhysics.calculate_owrk(m_list)
                    
                    if total_sfe:
                        st.info(f"### OWRK SFE Result")
                        sf1, sf2, sf3 = st.columns(3)
                        sf1.metric(R["lbl_total_sfe"], f"{total_sfe:.2f} mN/m")
                        sf2.metric(R["lbl_disperse"], f"{gamma_d:.2f} mN/m")
                        sf3.metric(R["lbl_polar"], f"{gamma_p:.2f} mN/m")
                
                # Export Buttons
                st.write("---")
                e1, e2 = st.columns(2)
                with e1:
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(R["btn_export_csv"], csv, "measurement_results.csv", "text/csv")
                with e2:
                    json_str = df.to_json(orient='records')
                    st.download_button(R["btn_export_json"], json_str, "measurement_results.json", "application/json")
            
        else:
            st.error(R["err_homography"])

    else:
        st.error(R["err_no_coin"])
        st.image(image_rgb, caption=R["cap_input"], use_container_width=True)
