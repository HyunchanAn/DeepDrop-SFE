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

# --- Language Resources ---
LANG_RESOURCES = {
    'EN': {
        'page_title': "DeepDrop Analyzer",
        'settings': "Settings",
        'model_type': "AI Model Type",
        'loaded_msg': "Loaded: {}",
        'exp_setup': "Experimental Setup",
        'liquid_type': "Liquid Type",
        'water': "Water",
        'diiodo': "Diiodomethane",
        'eg': "Ethylene Glycol",
        'gly': "Glycerol",
        'form': "Formamide",
        'caption': "v1.0.0 | Powered by MobileSAM",
        'main_title': "DeepDrop Analyzer",
        'sub_title': "AI-Based Surface Free Energy Analysis System",
        'error_model': "⚠️ MobileSAM Model not found. Please place `mobile_sam.pt` in the `models/` directory.",
        'error_loading': "⚠️ Error loading model: {}",
        'section_setup': "1. Setup & Upload",
        'upload_label': "Upload Droplet Image",
        'original_image': "Original Image",
        'analyze_btn': "🚀 Analyze Droplet",
        'spinner_msg': "AI Segmenting & Profiling...",
        'analysis_complete': "Analysis Complete!",
        'section_results': "2. Analysis Results",
        'seg_caption': "AI Segmentation (Confidence: {:.2f})",
        'metric_angle': "Contact Angle",
        'metric_conf': "AI Confidence",
        'section_history': "3. Measurement History",
        'btn_clear': "Reset History",
        'header_owrk': "### Total Surface Energy (OWRK)",
        'metric_sfe': "Total SFE",
        'label_dispersive': "- Dispersive: {:.2f} mN/m",
        'label_polar': "- Polar: {:.2f} mN/m",
        'warning_owrk': "Need at least 2 different liquids to calculate SFE.",
        'table_liquid': "Liquid",
        'table_angle': "Angle (°)",
        'msg_added': "Added to history."
    },
    'KR': {
        'page_title': "DeepDrop 분석기",
        'settings': "설정",
        'model_type': "AI 모델 선택",
        'loaded_msg': "로드됨: {}",
        'exp_setup': "실험 설정",
        'liquid_type': "용매 선택",
        'water': "물 (Water)",
        'diiodo': "다이아이오도메탄 (Diiodomethane)",
        'eg': "에틸렌 글리콜 (Ethylene Glycol)",
        'gly': "글리세롤 (Glycerol)",
        'form': "포름아마이드 (Formamide)",
        'caption': "v1.0.0 | MobileSAM 기반",
        'main_title': "DeepDrop 분석기",
        'sub_title': "AI 기반 정밀 표면 자유 에너지 분석 시스템",
        'error_model': "⚠️ MobileSAM 모델을 찾을 수 없습니다. `models/` 폴더에 `mobile_sam.pt` 파일을 넣어주세요.",
        'error_loading': "⚠️ 모델 로딩 오류: {}",
        'section_setup': "1. 설정 및 이미지 업로드",
        'upload_label': "액적 이미지 업로드",
        'original_image': "원본 이미지",
        'analyze_btn': "🚀 액적 분석 시작",
        'spinner_msg': "AI가 분석 중입니다...",
        'analysis_complete': "분석 완료!",
        'section_results': "2. 분석 결과",
        'seg_caption': "AI 세그멘테이션 (신뢰도: {:.2f})",
        'metric_angle': "접촉각 (Contact Angle)",
        'metric_conf': "AI 신뢰도",
        'section_history': "3. 측정 기록 (Measurement History)",
        'btn_clear': "기록 초기화",
        'header_owrk': "### 총 표면 에너지 (OWRK)",
        'metric_sfe': "총 표면 에너지",
        'label_dispersive': "- 분산 성분 (Dispersive): {:.2f} mN/m",
        'label_polar': "- 극성 성분 (Polar): {:.2f} mN/m",
        'warning_owrk': "표면 에너지 계산을 위해 최소 2가지 이상의 용매 데이터가 필요합니다.",
        'table_liquid': "용매",
        'table_angle': "접촉각 (°)",
        'msg_added': "기록에 추가되었습니다."
    }
}

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

# --- Language Toggle (Top Right) ---
col_empty, col_lang = st.columns([6, 1])
with col_lang:
    language = st.radio("Language", ["KR", "EN"], horizontal=True, label_visibility="collapsed")

R = LANG_RESOURCES[language]

# --- Sidebar ---
with st.sidebar:
    st.image("https://via.placeholder.com/300x100?text=DeepDrop+Logo", use_container_width=True)
    st.title(R['settings'])
    
    model_type = st.selectbox(R['model_type'], ["vit_t (MobileSAM)", "vit_h (Heavy)"])
    st.info(R['loaded_msg'].format(model_type))
    
    st.divider()
    
    st.subheader(R['exp_setup'])
    
    # Liquid type selection mapping
    liquid_map = {
        R['water']: "Water",
        R['diiodo']: "Diiodomethane",
        R['eg']: "Ethylene Glycol",
        R['gly']: "Glycerol",
        R['form']: "Formamide"
    }
    
    liquid_selection = st.selectbox(R['liquid_type'], list(liquid_map.keys()))
    current_liquid_internal = liquid_map[liquid_selection]
    
    st.divider()
    
    # History Management
    if 'measurements' not in st.session_state:
        st.session_state.measurements = []
        
    if st.button(R['btn_clear']):
        st.session_state.measurements = []
        st.rerun()
        
    st.caption(R['caption'])

# --- Main Content ---
st.markdown(f'<div class="main-header">{R["main_title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">{R["sub_title"]}</div>', unsafe_allow_html=True)

if analyzer is None:
    st.error(R['error_model'])
    st.stop()
elif isinstance(analyzer, str): # Error message
    st.error(R['error_loading'].format(analyzer))
    st.stop()

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(R['section_setup'])
    uploaded_file = st.file_uploader(R['upload_label'], type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        # Load Image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        st.image(image_rgb, caption=R['original_image'], use_container_width=True)
        
        if st.button(R['analyze_btn'], type="primary"):
            with st.spinner(R['spinner_msg']):
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
                    
                    # Add to history (prevent duplicates for same liquid if needed, but allowing override here)
                    # Simple append
                    st.session_state.measurements.append({
                        "liquid": current_liquid_internal,
                        "angle": contact_angle
                    })
                    
                    st.success(R['analysis_complete'])
                    st.toast(R['msg_added'])
                    
                    with col2:
                        st.subheader(R['section_results'])
                        st.image(res_img, caption=R['seg_caption'].format(score), use_container_width=True)
                        
                        # Metrics (Current)
                        m1, m2 = st.columns(2)
                        with m1:
                            st.metric(R['metric_angle'], f"{contact_angle:.2f}°")
                        with m2:
                            st.metric(R['metric_conf'], f"{score:.1%}")

# Always show History and SFE Calculation
with col2:
    if st.session_state.measurements:
        st.divider()
        st.subheader(R['section_history'])
        
        # Display Table
        history_data = [
            {R['table_liquid']: m['liquid'], R['table_angle']: f"{m['angle']:.2f}"} 
            for m in st.session_state.measurements
        ]
        st.table(history_data)
        
        # OWRK Calculation
        st.divider()
        st.markdown(R['header_owrk'])
        
        # Calculate SFE using all history
        total_sfe, sd, sp = DropletPhysics.calculate_owrk(st.session_state.measurements)
        
        if total_sfe is not None:
            st.metric(R['metric_sfe'], f"{total_sfe:.2f} mN/m")
            st.progress(min(total_sfe/100, 1.0))
            st.write(R['label_dispersive'].format(sd))
            st.write(R['label_polar'].format(sp))
        else:
            st.warning(R['warning_owrk'])

