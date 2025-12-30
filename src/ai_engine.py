import numpy as np
import torch
from mobile_sam import sam_model_registry, SamPredictor
import cv2
import os

class AICntactAngleAnalyzer:
    """
    MobileSAM 기반 액적 세그멘테이션 분석기
    """
    def __init__(self, model_path, model_type="vit_t", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading AI Model ({model_type}) on {self.device}...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}\nPlease download mobile_sam.pt and place it in the models directory.")
            
        self.sam = sam_model_registry[model_type](checkpoint=model_path)
        self.sam.to(device=self.device)
        self.sam.eval()
        self.predictor = SamPredictor(self.sam)
        print("Model loaded successfully.")

    def set_image(self, image_rgb):
        """
        MobileSAM Predictor에 이미지를 설정합니다.
        image_rgb: numpy array (H, W, 3) format
        """
        self.predictor.set_image(image_rgb)

    def predict_mask(self, points=None, labels=None, box=None):
        """
        프롬프트(점, 박스)를 기반으로 마스크를 생성합니다.
        기본적으로 이미지 중앙을 클릭하는 것으로 가정하여 Zero-shot 추론을 시도할 수 있습니다.
        """
        if points is None and box is None:
            # 기본 전략: 이미지 중앙 하단을 액적의 위치로 가정
            # (액적 접촉각 측정 이미지는 보통 중앙에 액적이 위치함)
             # image embedding shape을 가져올 수 없으므로 원본 이미지 사이즈를 알아야 함.
             # predictor.set_image를 호출했을 때의 shape을 사용.
             h, w = self.predictor.original_size
             input_point = np.array([[w // 2, h // 2]])
             input_label = np.array([1]) # 1: Foreground
             
             masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False, # 가장 신뢰도 높은 마스크 1개만
            )
        else:
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=box,
                multimask_output=False,
            )
            
        return masks[0], scores[0]

    def get_binary_mask(self, mask):
        """
        Boolean 마스크를 0/255 uint8 이미지로 변환합니다.
        """
        return (mask * 255).astype(np.uint8)
