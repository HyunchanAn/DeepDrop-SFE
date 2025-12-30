# DeepDrop-Analyzer (v1.0)
**AI 기반 정밀 표면 에너지 분석 솔루션 (AI-Based Surface Energy Analysis Solution)**

![AI Demo Placeholder](https://via.placeholder.com/800x400?text=AI+Segmentation+Demo+GIF+Here)
*(시연용: AI가 액적을 실시간으로 인식하고 분석하는 애니메이션이 이곳에 들어갑니다)*

## 1. 프로젝트 개요
**DeepDrop-Analyzer**는 기존의 단순 OpenCV 알고리즘의 한계(빛 반사, 노이즈 취약성)를 극복하기 위해, **최신 AI Segmentation 기술(MobileSAM)**과 **정밀 물리 엔진(Classical Fitting)**을 결합한 하이브리드 솔루션입니다.
학회 발표 및 대표이사 시연을 목적으로 엔터프라이즈급의 안정성과 시각적 탁월함을 제공합니다.

### 핵심 가치 (Key Value)
- **Zero-shot Recognition**: 학습되지 않은 새로운 액체나 복잡한 배경에서도 AI가 즉각적으로 액적의 경계를 인식합니다.
- **Robustness**: 딥러닝 마스킹을 통해 반사광(Reflection)이나 노이즈 문제를 완벽하게 필터링합니다.
- **Automation**: 사람의 개입 없이 Baseline을 AI가 스스로 설정하여 측정 오차를 최소화합니다.

## 2. 기술 스택 (Tech Stack)
본 프로젝트는 **Python** 기반의 웹 애플리케이션으로 구성되어 있습니다.

| 구분 | 기술 스택 | 비고 |
|---|---|---|
| **Core AI** | **MobileSAM (Segment Anything Model)** | 경량화된 Zero-shot Segmentation 모델 |
| **Physics** | **NumPy, SciPy, OpenCV** | 타원 피팅(Ellipse Fitting) 및 OWRK 수리 모델 연산 |
| **Frontend** | **Streamlit** | 실시간 데이터 시각화 및 인터랙티브 대시보드 |
| **Infra** | **Local (GPU/CPU)** | RTX 3060 이상 권장 (CPU 환경에서도 구동 가능) |

## 3. 설치 및 실행 방법

### 사전 요구 사항
- Python 3.8 이상
- (권장) CUDA 지원 GPU

### 설치
```bash
# 1. 저장소 클론
git clone https://github.com/your-repo/DeepDrop-SFE.git
cd DeepDrop-SFE

# 2. 의존성 설치
pip install -r requirements.txt

# 3. MobileSAM 가중치 다운로드
# 아래 링크에서 mobile_sam.pt를 다운로드 하여 models/ 폴더에 위치시켜야 합니다.
# https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
```

### 실행
```bash
streamlit run demo/app.py
```

## 4. 데이터셋 및 분석 파이프라인
1. **Image Input**: 실험 이미지 업로드 (JPG/PNG).
2. **AI Segmentation**: MobileSAM이 액적 영역을 정밀 마스킹.
3. **Mathematical Profiling**: 마스크 경계 추출 -> 타원 피팅 -> 접촉각(θ) 계산.
4. **SFE Computation**: 2가지 용매(물, Diiodomethane 등)의 접촉각을 통해 표면 에너지 산출 (OWRK 모델).

## 5. 라이선스
이 프로젝트는 **MIT License**를 따릅니다. MobileSAM 및 기타 오픈소스 라이브러리의 라이선스를 준수합니다.
