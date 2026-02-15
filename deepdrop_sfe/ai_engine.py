import numpy as np
import torch
from sam2.build_sam import build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import os

class AIContactAngleAnalyzer:
    """
    SAM2 (Segment Anything Model 2) 기반의 액적 및 참조 물체 분석기.
    RTX 5080과 같은 하이엔드 하드웨어에서 SAM 2.1 Large 모델을 사용하도록 최적화됨.
    """
    def __init__(self, model_id="facebook/sam2.1-hiera-large", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"SAM 2.1 모델 ({model_id})을 GPU에서 로드 중: {gpu_name} ({vram_total:.1f}GB VRAM)...")
            # 하이엔드 하드웨어 최적화 활성화
            torch.backends.cudnn.benchmark = True
        else:
            print(f"SAM 2.1 모델 ({model_id})을 {self.device}에서 로드 중...")

        # build_sam2_hf는 가중치 다운로드 및 설정을 자동으로 처리함
        try:
            self.model = build_sam2_hf(model_id, device=self.device)
            self.predictor = SAM2ImagePredictor(self.model)
            print(f"SAM 2.1 ({model_id}) 로드 완료.")
        except Exception as e:
            print(f"HF를 통한 SAM 2.1 모델 로드 실패: {e}")
            print("설정/가중치가 있는 경우 로컬 빌드로 대체를 시도합니다...")
            # 필요한 경우 대비책 로직을 여기에 추가할 수 있으나, 일반적으로 build_sam2_hf가 권장됨
            raise e

    def set_image(self, image_rgb):
        """
        SAM2 예측기를 위해 이미지를 설정함.
        image_rgb: numpy 배열 (H, W, 3) 형식
        """
        self.image_size = image_rgb.shape[:2]  # (H, W) 크기 저장
        self.predictor.set_image(image_rgb)

    def predict_mask(self, points=None, labels=None, box=None):
        """
        프롬프트(점, 박스)를 기반으로 마스크를 생성함.
        SAM2의 멀티 마스크 출력 및 필터링 로직을 사용함.
        """
        if not hasattr(self, "image_size"):
             raise RuntimeError("predict_mask() 호출 전에 set_image()를 먼저 실행하십시오.")

        h, w = self.image_size

        if points is None and box is None:
             # Default: Center point strategy
             input_point = np.array([[w // 2, h // 2]])
             input_label = np.array([1]) 
             
             masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
        else:
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=box,
                multimask_output=True,
            )
            
        # 가장 적합한 마스크를 선택함.
        valid_masks = []
        img_area = h * w
        
        for i in range(len(masks)):
            mask = masks[i]
            mask_area = np.sum(mask)
            if mask_area < 10 or mask_area > 0.8 * img_area:
                continue
            
            # 경계 영역 체크
            edge_pixels = np.sum(mask[0, :]) + np.sum(mask[-1, :]) + np.sum(mask[:, 0]) + np.sum(mask[:, -1])
            edge_ratio = edge_pixels / (2 * (mask.shape[0] + mask.shape[1]))
            
            final_score = scores[i]
            if edge_ratio > 0.05:
                final_score *= 0.1
            
            if box is not None:
                x1, y1, x2, y2 = box
                box_mask = mask[int(y1):int(y2), int(x1):int(x2)]
                contained_area = np.sum(box_mask)
                containment_ratio = contained_area / mask_area if mask_area > 0 else 0
                final_score *= (0.3 + 0.7 * containment_ratio)

            valid_masks.append({
                'mask': mask,
                'sam_score': scores[i],
                'final_score': final_score,
                'area': mask_area
            })
            
        if not valid_masks:
            idx = np.argmax(scores)
            return masks[idx], scores[idx]
            
        best = max(valid_masks, key=lambda x: x['final_score'])
        
        # --- Post-processing: Clean Mask ---
        final_mask = self.clean_mask(best['mask'])
        
        return final_mask, best['sam_score']

    def clean_mask(self, mask):
        """
        Applies morphological operations to remove small noise and smooth edges.
        """
        mask_uint8 = (mask * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Remove small noise dots (Opening)
        mask_clean = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        # Fill small holes inside (Closing)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
        
        return mask_clean > 127

    def auto_detect_coin_candidate(self, image_cv2):
        """
        CLAHE 전처리, 가중치 기반 필터링을 사용하여 사선 방향의 동전 후보군을 정밀하게 찾음.
        """
        gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        img_area = h * w
        
        # --- 1. 전처리 강화 (Preprocessing) ---
        # CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 가우시안 블러로 미세 노이즈 억제 (배경 나뭇결 무늬 등)
        blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)
        
        candidates = []

        # --- 2. 전략 1: 허프 원 변환 (Hough Circle) ---
        # 파라미터를 다소 엄격하게 조정하여 확실한 원형만 탐색
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=h/10,
                                   param1=120, param2=35, minRadius=int(h*0.1), maxRadius=int(h*0.4))
        if circles is not None:
             circles = circles[0, :]
             for cx, cy, r in circles:
                 # 이미지 경계를 벗어나는지 체크
                 if cx - r < 0 or cy - r < 0 or cx + r > w or cy + r > h:
                     continue
                 
                 candidates.append({
                     'box': [max(0, cx-r-5), max(0, cy-r-5), min(w, cx+r+5), min(h, cy+r+5)],
                     'score': 0.9, # 허프 변환 결과는 기본적으로 높은 점수 부여
                     'area': np.pi * (r**2),
                     'method': 'Hough'
                 })

        # --- 3. 전략 2 & 3: 컨투어 분석 (Contour Analysis) ---
        def process_contours(binary_img, method_name):
            cnts, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                
                # 면적 필터링: 전체 이미지의 1% ~ 40% 사이여야 함 (사선 촬영 시 면적이 작게 보일 수도, 크게 보일 수도 있음)
                area_ratio = area / img_area
                if area_ratio < 0.01 or area_ratio > 0.40: 
                    continue

                # 볼록 껍질(Hull) 분석
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0: continue
                
                hull_perimeter = cv2.arcLength(hull, True)
                if hull_perimeter == 0: continue
                
                # 지표 계산
                solidity = float(area) / hull_area
                # 원형도 계산 (4 * pi * area / perimeter^2)
                circularity = 4 * np.pi * hull_area / (hull_perimeter * hull_perimeter)
                
                # 종횡비(Aspect Ratio) 확인 - 사선 촬영 고려
                if len(hull) >= 5:
                    ellipse = cv2.fitEllipse(hull)
                    (xc, yc), (d1, d2), angle = ellipse
                    ar = max(d1, d2) / (min(d1, d2) + 1e-6)
                else:
                    ar = 100
                
                # 필터링 조건 강화 (Oblique Friendly):
                # 1. 어느 정도 볼록해야 함 (Solidity > 0.85)
                # 2. 사선으로 찍혀도 타원 형태를 유지해야 함 (AR < 4.0)
                # 3. 모양이 아주 망가지지 않아야 함 (Circularity > 0.4)
                if solidity < 0.85 or ar > 4.0 or circularity < 0.4: 
                    continue
                
                # 가중 점수 계산: (원형도 * 0.3) + (솔리디티 * 0.4) + (면적 비율 * 0.3)
                # 사선 촬영 시 원형도는 낮아지므로 가중치를 조금 낮춤
                score = (circularity * 0.3) + (solidity * 0.4) + (min(1.0, area_ratio/0.1) * 0.3)
                
                x, y, bw, bh = cv2.boundingRect(hull)
                candidates.append({
                    'box': [max(0, x-5), max(0, y-5), min(w, x+bw+5), min(h, y+bh+5)],
                    'score': score,
                    'area': area,
                    'method': method_name
                })

        # 오츠 이진화 (Otsu) 전에 적응형 임계값 한 번 더 사용
        thresh_adapt = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 15, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        thresh_adapt = cv2.morphologyEx(thresh_adapt, cv2.MORPH_OPEN, kernel)
        process_contours(thresh_adapt, 'Adaptive')
        
        _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel)
        process_contours(thresh_otsu, 'Otsu')

        if not candidates:
            print("동전 후보군을 찾지 못했습니다. (이미지 대비 혹은 조도가 원인일 수 있음)")
            return None
            
        # 가중치 점수 기반으로 가장 적합한 객체 선택
        best = max(candidates, key=lambda x: x['score'])
        print(f"[{best['method']}] 방식을 통해 동전을 성공적으로 감지했습니다. (점수: {best['score']:.2f}, 이미지 대비 면적: {(best['area']/img_area)*100:.1f}%)")
        
        return np.array(best['box'])

    def auto_detect_droplet_candidate(self, image_cv2, exclude_box=None):
        """
        기본적인 컴퓨터 비전 기법을 사용하여 액적 후보군을 찾음.
        exclude_box: 무시할 영역 [x1, y1, x2, y2] (예: 동전 영역)
        """
        gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 작고 원형에 가까운 물체를 탐색
        # 액적은 보통 국부 대비가 높음
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 15, 5)
        
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        img_area = image_cv2.shape[0] * image_cv2.shape[1]
        candidates = []
        
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 100 or area > img_area * 0.1: # Too small or too large
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 이 박스가 exclude_box(동전)과 겹치는지 확인
            if exclude_box is not None:
                ex1, ey1, ex2, ey2 = exclude_box
                # 교집합 영역 계산
                ix1 = max(x, ex1)
                iy1 = max(y, ey1)
                ix2 = min(x+w, ex2)
                iy2 = min(y+h, ey2)
                if ix1 < ix2 and iy1 < iy2:
                    intersect_area = (ix2 - ix1) * (iy2 - iy1)
                    if intersect_area > 0.5 * area:
                        continue
            
            # 지표: Solidity 및 Circularity
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0: continue
            solidity = float(area) / hull_area
            
            peri = cv2.arcLength(cnt, True)
            if peri == 0: continue
            circularity = 4 * np.pi * area / (peri * peri)
            
            # 액적은 일반적으로 볼록(Convex)하며 어느 정도 원형을 유지함
            if solidity > 0.8 and circularity > 0.5:
                candidates.append({
                    'box': [max(0, x-5), max(0, y-5), min(image_cv2.shape[1], x+w+5), min(image_cv2.shape[0], y+h+5)],
                    'area': area,
                    'score': solidity * circularity
                })
        
        if not candidates:
            return None
            
        # 가장 적합한 후보군 선택 (solidity * circularity 점수가 가장 높은 것)
        best = max(candidates, key=lambda x: x['score'])
        return np.array(best['box'])

    def get_binary_mask(self, mask):
        """
        불리언 마스크를 uint8 0/255 형식으로 변환함.
        """
        return (mask * 255).astype(np.uint8)
