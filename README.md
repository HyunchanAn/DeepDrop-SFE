# DeepDrop-AnyView (v2.1)
**Arbitrary-Angle Surface Energy & Contact Angle Analysis System**

![DeepDrop AnyView](https://via.placeholder.com/800x400?text=DeepDrop+AnyView+Demo)

DeepDrop-AnyView는 기존 측면(Side-view) 촬영 방식의 제약을 극복한 임의 각도(Arbitrary-view) 접촉각 분석 시스템입니다. 100원 동전과 같은 참조 물체(Reference Object)를 활용하여 촬영 각도에 따른 원근 왜곡을 자동으로 보정(Homography)하며, 액적의 실제 부피 기반 물리 모델을 통해 정밀한 접촉각을 산출합니다.

## 주요 기능 (Key Features)

### 1. Arbitrary View Analysis (임의 각도 분석)
- **제약 해소**: 수평 90도 촬영이 어려운 환경에서도 45~80도 사이의 사선 각도 촬영 이미지를 분석할 수 있습니다.
- **원근 보정**: **Homography** 기술을 적용하여 사선 이미지를 평면(Top-down) 뷰로 재구성하여 분석 오차를 최소화합니다.

### 2. Intelligent Reference Calibration (지능형 참조 물체 보정)
- **자동 감지**: 100원 동전(구형/신형) 및 500원, 10원 동전을 자동으로 인식하여 **Pixel-to-mm** 스케일을 즉시 계산합니다.
- **수동 보정 (Manual Fallback)**: 조명이 열악해 AI가 물체를 찾지 못할 경우, 사용자가 직접 영역을 지정할 수 있는 **Canvas Drawing** 기능을 제공합니다 (모바일 터치 지원).

### 3. Physics-Based Volume Logic (물리 기반 부피 연산)
- **정밀 알고리즘**: 단순한 타원 피팅 방식이 아닌, 액적의 실제 **부피(Volume)**와 **접촉 반경(Contact Radius)**을 이용한 수치 해석(Numerical Solver) 모델을 사용합니다.
- **실시간 프로필 시각화**: 산출된 접촉각을 기반으로 물방울의 측면 프로필을 실시간 생성하여 분석 결과의 시각적 검증이 가능합니다.

### 4. High-End Hardware Optimization (하드웨어 최적화)
- **NVIDIA RTX 5080 지원**: 최신 Blackwell 아키텍처 및 CUDA 12.8 환경에서 **SAM 2.1 (Segment Anything Model 2.1)** Large 모델을 구동하여 초고속 이미지 분석을 제공합니다.

---

## 기술 스택 (Tech Stack)

| 구성 요소 | 기술 | 설명 |
|---|---|---|
| **AI Engine** | **SAM 2.1 (Large)** | 고정밀 세그멘테이션 (RTX 5080 기반 최적화) |
| **CV Engine** | **OpenCV** | Perspective Correction & Contour Analysis |
| **Physics** | **SciPy Optimization** | 부피 및 직경 기반 접촉각 역산 (OWRK 모델 포함) |
| **Frontend** | **Streamlit** | 대화형 UI (모바일/PC 하이브리드 대응) |

---

## 설치 및 실행 (Installation & Run)

### 1. 환경 요구 사양
- **Python**: 3.12 ~ 3.14 권장
- **GPU**: NVIDIA RTX 40/50 시리즈 (CUDA 12.x 필수)
- **최적화**: RTX 5080 사용자의 경우 CUDA 12.8 기반 PyTorch 나이틀리 빌드를 권장합니다.

### 2. 설치
```bash
# 저장소 복제
git clone https://github.com/HyunchanAn/SG_proj_002.git
cd SG_proj_002

# 의존성 설치
pip install -r requirements.txt
```

### 3. 모델 가중치
프로젝트 실행 시 Hugging Face를 통해 `facebook/sam2.1-hiera-large` 모델이 자동으로 다운로드 및 캐싱됩니다.

### 4. 실행
```bash
python -m streamlit run demo/app.py
```

---

## 라이브러리 사용법 (Library Usage)

본 프로젝트의 엔진은 **`deepdrop_sfe`** 패키지로 모듈화되어 있어 외부 프로젝트에서도 쉽게 연동 가능합니다.

```python
from deepdrop_sfe import DropletPhysics

# 1. 접촉각 계산
# 입력: 부피(µL), 접촉 직경(mm)
angle, diag = DropletPhysics.calculate_contact_angle(3.0, 9.13, return_info=True)
print(f"Contact Angle: {angle:.2f}°")

# 2. 표면 에너지(SFE) 분석 (OWRK)
data = [
    {'liquid': 'Water', 'angle': 110.0},
    {'liquid': 'Diiodomethane', 'angle': 45.0}
]
sfe, disperse, polar = DropletPhysics.calculate_owrk(data)
print(f"Surface Energy: {sfe:.2f} mN/m")
```

---

## 촬영 및 분석 가이드 (Guide)

1. **동전 배치**: 액적과 평행한 위치에 **100원 동전**을 놓고 가급적 선명하게 촬영하세요.
2. **배경 조건**: 반사가 적고 매끄러운 단색 배경이 세그멘테이션 품질에 유리합니다.
3. **권장 각도**: 이미지 상단에서 비스듬히 내려다보는 **45도 ~ 80도** 각도가 보정에 가장 적합합니다.

---

## 라이선스 및 출처 (License & Attribution)
- 본 프로젝트는 **MIT License**를 따릅니다.
- 세그멘테이션 엔진으로 [SAM 2.1 (facebookresearch/segment-anything-2)](https://github.com/facebookresearch/segment-anything-2)을 사용합니다.
- (과거 버전에서 MobileSAM을 사용하였으나, 현재는 SAM 2.1로 완전히 전환되었습니다.)
