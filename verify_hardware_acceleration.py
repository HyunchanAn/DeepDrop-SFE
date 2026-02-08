import torch
import time
import numpy as np
import os
from deepdrop_sfe.ai_engine import AIContactAngleAnalyzer
import cv2

def verify_system():
    print("="*50)
    print("DeepDrop-SFE 하드웨어 가속 검증")
    print("="*50)
    
    # 1. CPU 및 OS 확인
    import platform
    print(f"[OS] {platform.system()} {platform.release()}")
    print(f"[CPU] {platform.processor()}")
    
    # 2. GPU 확인
    cuda_available = torch.cuda.is_available()
    print(f"[GPU] CUDA 사용 가능: {cuda_available}")
    if cuda_available:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  - 장치 {i}: {props.name}")
            print(f"    VRAM: {round(props.total_memory / (1024**3), 1)} GB")
            print(f"    연산 능력(Compute Capability): {props.major}.{props.minor}")
    
    # 3. SAM 2.1 추론 테스트
    print("\n[AI] 검증을 위해 SAM 2.1 (facebook/sam2.1-hiera-tiny) 초기화 중...")
    start_time = time.time()
    try:
        analyzer = AIContactAngleAnalyzer(model_id="facebook/sam2.1-hiera-tiny")
        load_duration = time.time() - start_time
        print(f"  - 모델 로드 시간: {load_duration:.2f}초")
        
        # 추론 테스트를 위한 가상 이미지 생성
        test_img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        
        print("  - 추론 실행 중 (1024x1024 이미지)...")
        start_time = time.time()
        analyzer.set_image(test_img)
        set_img_duration = time.time() - start_time
        
        start_time = time.time()
        result = analyzer.predict_mask()
        inference_duration = time.time() - start_time
        
        print(f"  - predict_mask 결과 타입: {type(result)}")
        if isinstance(result, tuple):
            print(f"  - 결과 길이: {len(result)}")
            mask, score = result
        else:
            mask = result
            score = 0
            print("  - 경고: predict_mask가 튜플 대신 단일 값을 반환했습니다.")
        
        print(f"  - set_image 시간: {set_img_duration * 1000:.1f}ms")
        print(f"  - predict_mask 시간: {inference_duration * 1000:.1f}ms")
        print(f"  - 예측 점수(Prediction Score): {score:.4f}")
        
        if cuda_available:
            max_mem = torch.cuda.max_memory_allocated() / (1024**2)
            print(f"  - 최대 GPU 메모리 점유: {max_mem:.1f} MB")
            
        print("\n[상태] SAM2.1 하드웨어 가속: 성공 ✅")
        
    except Exception as e:
        print(f"\n[상태] SAM2.1 통합 테스트: 실패 ❌")
        print(f"오류 내용: {e}")

if __name__ == "__main__":
    verify_system()
