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

    def predict_mask(self, point_coords=None, point_labels=None, box=None, prefer_largest=False, prefer_circular=False):
        """
        프롬프트(점, 박스)를 기반으로 마스크를 생성함.
        SAM2의 멀티 마스크 출력 및 필터링 로직을 사용함.
        """
        if not hasattr(self, "image_size"):
             raise RuntimeError("predict_mask() 호출 전에 set_image()를 먼저 실행하십시오.")

        h, w = self.image_size

        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
            
        valid_masks = []
        img_area = h * w
        
        for i in range(len(masks)):
            mask = masks[i]
            mask_area = np.sum(mask)
            # 면적 범위: 전체의 1% ~ 50%
            if mask_area < img_area * 0.01 or mask_area > 0.5 * img_area:
                continue
            
            mask_uint8 = (mask * 255).astype(np.uint8)
            cnts, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: continue
            
            best_cnt = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(best_cnt)
            perimeter = cv2.arcLength(best_cnt, True)
            if perimeter == 0: continue
            
            # 원형도 점수 (Circularity: 4*pi*area / perimeter^2)
            circularity = (4 * np.pi * area) / (perimeter * perimeter)
            
            # 기본 점수는 SAM의 Confidence
            final_score = scores[i]
            
            # 원형도 가중치 강화 (동전 탐지 시 최우선)
            if prefer_circular:
                # 원형도가 0.85 이상이면 매우 높은 가중치 부여
                circ_weight = 0.8 if circularity > 0.85 else circularity
                final_score = final_score * 0.3 + circ_weight * 0.7
            
            # 면적 가중치 (Large 객체 선호 시)
            if prefer_largest:
                size_score = min(1.0, mask_area / (0.10 * img_area))
                final_score = final_score * 0.4 + size_score * 0.6
            
            # 박스 제약 조건 완화: 원형도가 높으면 박스를 조금 벗어나더라도 점수 유지
            if box is not None:
                bx1, by1, bx2, by2 = box
                mx, my, mw_m, mh_m = cv2.boundingRect(best_cnt)
                # 박스 이탈 허용 범위: 박스 크기의 20%까지
                bw_p = (bx2 - bx1) * 0.2
                bh_p = (by2 - by1) * 0.2
                if mx < bx1 - bw_p or my < by1 - bh_p or mx + mw_m > bx2 + bw_p or my + mh_m > by2 + bh_p:
                    if circularity < 0.90: # 원형도가 낮으면서 박스를 이탈하면 점수 삭감
                        final_score *= 0.3

            valid_masks.append({
                'mask': mask,
                'sam_score': scores[i],
                'final_score': final_score,
                'area': mask_area,
                'circularity': circularity
            })
            
        if not valid_masks:
            # 유효한 마스크가 없으면 가장 점수 높은 마스크라도 반환
            idx = np.argmax(scores)
            return masks[idx], scores[idx]
            
        # 원형도와 면적이 최적인 마스크 선택
        best = max(valid_masks, key=lambda x: x['final_score'])
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

    def _detect_coin_by_features(self, image_cv2, template_path="coin_10krw_template.png"):
        """
        SIFT 특징점 매칭을 사용하여 이미지 내에서 템플릿(동전)과 가장 유사한 영역을 찾음.
        """
        if not os.path.exists(template_path):
            return None
        
        try:
            template = cv2.imread(template_path, 0)
            target = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
            
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(template, None)
            kp2, des2 = sift.detectAndCompute(target, None)
            
            if des1 is None or des2 is None: return None
            
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            
            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)
            
            # 매칭 점이 최소 8개 이상이어야 신뢰 가능
            if len(good) >= 8:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    h_t, w_t = template.shape
                    pts = np.float32([[0, 0], [0, h_t-1], [w_t-1, h_t-1], [w_t-1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    
                    x_coords = dst[:, 0, 0]
                    y_coords = dst[:, 0, 1]
                    box = [float(np.min(x_coords)), float(np.min(y_coords)), float(np.max(x_coords)), float(np.max(y_coords))]
                    center = [float(np.mean(x_coords)), float(np.mean(y_coords))]
                    return {'box': box, 'center': center, 'score': len(good) / 50.0}
        except Exception as e:
            print(f"[Feature Match Error] {e}")
            
        return None

    def auto_detect_coin_candidate(self, image_cv2):
        """
        원주 우선(Rim-first) 전략을 사용하여 동전 내부 문양을 배제하고 전체 테두리를 감지함.
        Hough Circle의 물리적 제약 강화 + 엣지 팽창(Dilation) 기반 센터 추정 사용.
        """
        image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        h, w = image_cv2.shape[:2]
        gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        
        best_cx, best_cy, best_r = 0.0, 0.0, 0.0
        found = False
        method_name = ""
        
        candidates = []
        
        # --- 1. 경로 A: 타원 탐색 (Precision-focused) ---
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.medianBlur(enhanced, 5) 
        edges = cv2.Canny(denoised, 35, 90)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if len(cnt) < 5: continue
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if not (h * w * 0.005 < area < h * w * 0.35): continue
            
            ellipse = cv2.fitEllipse(cnt)
            (cx_e, cy_e), (ma, Mi), angle = ellipse
            
            aspect_ratio = max(ma, Mi) / (min(ma, Mi) + 1e-6)
            if aspect_ratio > 1.8: continue
            
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-6)
            if solidity < 0.8: continue
            
            # 가중치 스코어: 원형도와 면적, 중앙 집중도 결합
            dist_to_center = np.sqrt((cx_e-w/2)**2 + (cy_e-h/2)**2)
            score = (area * circularity) * (1.1 - dist_to_center/(w/2))
            
            candidates.append({
                'center': (cx_e, cy_e), 
                'radius': max(ma, Mi)/2, 
                'score': score, 
                'method': 'Precision-Ellipse',
                'geom': ellipse
            })
            
        # --- 2. 경로 B: 허프 원 변환 (Fallback) ---
        blurred_h = cv2.medianBlur(gray, 7)
        circles = cv2.HoughCircles(blurred_h, cv2.HOUGH_GRADIENT, 1.2, minDist=h/8,
                                  param1=100, param2=22, minRadius=int(h*0.08), maxRadius=int(h*0.35))
        
        if circles is not None:
            for c in circles[0, :]:
                dist_to_center = np.sqrt((c[0]-w/2)**2 + (c[1]-h/2)**2)
                # 허프 원은 그 자체로 원형도가 보장되므로 면적과 위치만 고려
                score = (np.pi * (c[2]**2)) * (1.0 - dist_to_center/(w/2))
                candidates.append({
                    'center': (c[0], c[1]), 
                    'radius': c[2], 
                    'score': score, 
                    'method': 'Anchor-Hough'
                })

        if not candidates:
            print("[Auto-Detect] 동전 후보 탐색 실패.")
            return None
            
        # 가장 그럴듯한(원형이면서 적절한 크기의 중앙 객체) 후보 선택
        best = max(candidates, key=lambda x: x['score'])
        best_cx, best_cy, best_r = float(best['center'][0]), float(best['center'][1]), float(best['radius'])
        method_name = best['method']
        found = True
        
        print(f"[{method_name}] 최종 후보 결정: ({best_cx:.1f}, {best_cy:.1f}), R: {best_r:.1f}")
        
        # 가이드 박스 생성
        best_box = [best_cx - best_r*1.1, best_cy - best_r*1.1, best_cx + best_r*1.1, best_cy + best_r*1.1]

        if not found:
            print("[Auto-Detect] 동전 타원/원형 검출 실패.")
            return None
            
        print(f"[{method_name}] 동전 위치 확정: ({best_cx:.1f}, {best_cy:.1f}), R_max: {best_r:.1f}")

        # --- 3. 지능형 프롬프트 구성 (강화버전) ---
        cx, cy, r = best_cx, best_cy, best_r
        
        # 긍정 포인트: 중앙 및 테두리 안쪽 4곳
        pos_points = [[cx, cy]]
        for ang in [0, 90, 180, 270]:
            rad_val = np.deg2rad(ang)
            pos_points.append([cx + r*0.7*np.cos(rad_val), cy + r*0.7*np.sin(rad_val)])
            
        # 부정 포인트: 테두리 바깥 4곳 (배경 고립)
        neg_points = []
        for ang in [45, 135, 225, 315]:
            rad_val = np.deg2rad(ang)
            neg_points.append([cx + r*1.5*np.cos(rad_val), cy + r*1.5*np.sin(rad_val)])
            
        input_points = np.array(pos_points + neg_points)
        input_labels = np.array([1]*len(pos_points) + [0]*len(neg_points))
        
        # --- 4. SAM 호출 (Circular & Largest 전략) ---
        try:
            self.predictor.set_image(image_rgb)
            self.image_size = (h, w)
            
            # 박스 프롬프트는 Rim을 여유있게 포함하도록 설정
            box_prompt = np.array([cx - r*1.2, cy - r*1.2, cx + r*1.2, cy + r*1.2])
            
            refined_mask, _ = self.predict_mask(
                point_coords=input_points, 
                point_labels=input_labels, 
                box=box_prompt,
                prefer_largest=True,
                prefer_circular=True 
            )
            
            refined_mask_uint8 = (refined_mask * 255).astype(np.uint8)
            cnts, _ = cv2.findContours(refined_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if cnts:
                best_cnt = max(cnts, key=cv2.contourArea)
                rx, ry, rbw, rbh = cv2.boundingRect(best_cnt)
                padding = 5
                final_box = [max(0, rx-padding), max(0, ry-padding), min(w, rx+rbw+padding), min(h, ry+rbh+padding)]
                # Return both box and circle info for UI sliders
                return np.array(final_box), (float(cx), float(cy), float(r))
        except Exception as e:
            print(f"[Rim-SAM Error] {e}")
            
        return None, None

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
