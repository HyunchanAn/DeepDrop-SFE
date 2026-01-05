# 기본 이미지 설정 (안정적인 bookworm 기반 사용)
FROM python:3.10-slim-bookworm

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치 (OpenCV 등)
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    curl \
    software-properties-common \
    git \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 패키지 설치
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 프로젝트 소스 코드 및 모델 가중치 복사
COPY . .

# Streamlit 포트 노출
EXPOSE 8501

# 헬스체크 설정
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# 실행 명령어
ENTRYPOINT ["streamlit", "run", "demo/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
