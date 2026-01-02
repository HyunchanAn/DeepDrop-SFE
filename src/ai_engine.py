import numpy as np
import torch
from mobile_sam import sam_model_registry, SamPredictor
import cv2
import os

class AIContactAngleAnalyzer:
    """
    MobileSAM based Droplet and Reference Object Analyzer.
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
        Sets the image for the MobileSAM predictor.
        image_rgb: numpy array (H, W, 3) format
        """
        self.predictor.set_image(image_rgb)

    def predict_mask(self, points=None, labels=None, box=None):
        """
        Generates a mask based on prompts (points, box).
        """
        if points is None and box is None:
             # Default: Center point strategy
             h, w = self.predictor.original_size
             input_point = np.array([[w // 2, h // 2]])
             input_label = np.array([1]) 
             
             masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )
        else:
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=box,
                multimask_output=False,
            )
            
        mask = masks[0]
        score = scores[0]
        return mask, score

    def auto_detect_coin_candidate(self, image_cv2):
        """
        Uses OpenCV heuristics to find the most 'circular' object as a coin candidate.
        Returns the bounding box [x_min, y_min, x_max, y_max] or None.
        """
        gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        
        # Preprocessing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Adaptive Thresholding for better robustness against lighting
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_cnt = None
        best_score = -1
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500: # Filter small noise
                continue
                
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
                
            # Circularity = 4 * pi * Area / (Perimeter^2)
            # Perfect circle = 1.0
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Additional check: Convex Hull solidity
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0: continue
            solidity = float(area) / hull_area
            
            # Score: High circularity and high solidity
            score = circularity * solidity
            
            if score > best_score:
                best_score = score
                best_cnt = cnt
                
        if best_cnt is not None and best_score > 0.6: # Threshold for "Circle-like"
            x, y, w, h = cv2.boundingRect(best_cnt)
            # Add padding
            pad = 5
            x_min = max(0, x - pad)
            y_min = max(0, y - pad)
            x_max = min(image_cv2.shape[1], x + w + pad)
            y_max = min(image_cv2.shape[0], y + h + pad)
            return np.array([x_min, y_min, x_max, y_max])
            
        return None

    def get_binary_mask(self, mask):
        """
        Converts boolean mask to uint8 0/255.
        """
        return (mask * 255).astype(np.uint8)
