# DeepDrop-AnyView (v2.0)
**Arbitrary-Angle Surface Energy & Contact Angle Analysis System**

![DeepDrop AnyView](https://via.placeholder.com/800x400?text=DeepDrop+AnyView+Demo)

**DeepDrop-AnyView**는 기존 측면(Side-view) 촬영 방식의 제약을 없앤 **임의 각도(Arbitrary-view) 접촉각 분석 시스템**입니다.
10원 동전과 같은 **참조 물체(Reference Object)**를 이용하여 이미지를 **Top-view로 원근 보정(Homography)**하고, 물리적 부피(Volume) 기반으로 정확한 접촉각을 산출합니다.

## 주요 기능 (Key Features)

### 1. Arbitrary View Analysis (임의 각도 분석)
- **Problem**: 기존 시스템은 정확한 90도 측면 촬영이 필수였습니다.
- **Solution**: **Homography** 기술을 적용하여 어떤 각도에서 찍은 사진이든 평면(Top-down) 이미지로 변환하여 분석합니다.

### 2. Reference Object Calibration (참조 물체 기반 보정)
- **Auto Scale**: 10원 동전(구형/신형)을 함께 촬영하면, AI가 이를 자동으로 감지하여 **Pixel-to-mm** 스케일을 계산합니다.
- **Manual Fallback**: 조명이나 각도가 난해하여 AI가 동전을 찾지 못할 경우, **직접 그리기(Manual Selection)** 모드를 통해 분석을 계속할 수 있습니다.

### 3. Volume-Based Calculation (부피 기반 연산)
- **Physics Engine**: 단순한 타원 피팅이 아닌, 액적의 실제 **부피(Volume)**와 **접촉 반경(Contact Radius)**을 통해 물리적으로 타당한 접촉각을 역산(Numerical Solver)합니다.

---

## 기술 스택 (Tech Stack)

| Component | Technology | Description |
|---|---|---|
| **AI Engine** | **MobileSAM** | Zero-shot Segmentation (학습 없는 즉각적 액적/동전 인식) |
| **CV Engine** | **OpenCV Homography** | Perspective Correction (원근 보정 및 이미지 변환) |
| **Physics** | **SciPy Optimization** | Volume & Diameter based Contact Angle Calculation |
| **Frontend** | **Streamlit** | Interactive UI (Drag & Drop, Manual Drawing) |

---

## 설치 및 실행 (Installation & Run)

### 1. 환경 설정 (Prerequisites)
- Python 3.9+
- CUDA GPU 권장 (CPU 모드 지원)

### 2. 설치 (Installation)
```bash
# Clone Repository
git clone https://github.com/your-repo/DeepDrop-SFE.git
cd DeepDrop-SFE

# Install Dependencies
pip install -r requirements.txt
```

### 3. 모델 다운로드 (Model Weights)
`models/` 폴더에 MobileSAM 가중치 파일(`mobile_sam.pt`)이 있어야 합니다.
- [MobileSAM Weights Download Link](https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt)

### 4. 실행 (Run)
```bash
python -m streamlit run demo/app.py
```

---

## 촬영 가이드 (Photography Guide)
정확한 분석을 위해 다음 사항을 지켜주세요:

1. **동전 배치**: 액적 옆에 **10원 동전**을 놓고 함께 촬영하세요. (동전이 너무 멀리 있으면 초점이 안 맞을 수 있습니다.)
2. **배경**: 매끄러운 단색 배경이 가장 좋습니다. (반사가 심한 유리는 피하는 것이 좋습니다.)
3. **각도**: 너무 극단적인 각도(거의 수평)보다는 **45도~80도** 정도의 사선 각도가 가장 분석하기 좋습니다.

---

## License
This project is licensed under the **MIT License**.
Based on [MobileSAM](https://github.com/ChaoningZhang/MobileSAM).
