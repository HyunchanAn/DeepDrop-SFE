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
        Uses Hull Circularity and Solidity to find oblique coin candidates.
        """
        gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        rows = gray.shape[0]
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        candidates = []

        # Strategy 1: Hough Circle
        # param2 lowered to 20 to find more circles (weak ones)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=rows/10,
                                   param1=100, param2=25, minRadius=20, maxRadius=rows//2)
        if circles is not None:
             circles = np.uint8(np.around(circles))
             # Pick the largest circle by radius
             best_circle = max(circles[0, :], key=lambda x: x[2])
             cx, cy, r = best_circle
             
             # Check if this circle roughly matches edges? 
             # Trust Hough for now if it finds something circular.
             candidates.append({
                 'box': [max(0, cx-r-10), max(0, cy-r-10), min(image_cv2.shape[1], cx+r+10), min(image_cv2.shape[0], cy+r+10)],
                 'score': 0.95, 
                 'area': np.pi * r * r,
                 'method': 'Hough'
             })

        # Top-down logic for contours
        def process_contours(binary_img, method_name):
            cnts, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                if len(cnt) < 5: continue
                
                area = cv2.contourArea(cnt)
                if area < 500: continue

                # Hull Analysis
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0: continue
                
                hull_perimeter = cv2.arcLength(hull, True)
                if hull_perimeter == 0: continue
                
                # Metrics
                solidity = float(area) / hull_area
                hull_circularity = 4 * np.pi * hull_area / (hull_perimeter * hull_perimeter)
                
                # Fit Ellipse to check Aspect Ratio
                if len(hull) >= 5:
                    ellipse = cv2.fitEllipse(hull)
                    (xc, yc), (d1, d2), angle = ellipse
                    major = max(d1, d2)
                    minor = min(d1, d2)
                    ar = major / minor if minor > 0 else 100
                else:
                    ar = 100
                
                # Filter:
                # 1. Coin must be convex -> High Solidity
                if solidity < 0.85: continue
                
                # 2. Coin must be somewhat elliptical -> Aspect Ratio not crazy
                if ar > 4.5: continue
                
                # 3. Shape must be smooth (Ellipse/Circle-like) -> Hull Circularity
                # Perfect circle = 1.0. 
                # Very oblique ellipse (AR=3) -> Circ ~ 0.65
                if hull_circularity < 0.5: continue
                
                # Score logic
                score = solidity
                
                x, y, w, h = cv2.boundingRect(hull)
                pad = 10
                candidates.append({
                    'box': [max(0, x-pad), max(0, y-pad), min(image_cv2.shape[1], x+w+pad), min(image_cv2.shape[0], y+h+pad)],
                    'score': score,
                    'area': area,
                    'method': method_name
                })

        # Strategy 2: Adaptive Thresholding
        thresh_adapt = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3,3), np.uint8)
        thresh_adapt = cv2.morphologyEx(thresh_adapt, cv2.MORPH_CLOSE, kernel)
        process_contours(thresh_adapt, 'Adaptive')
        
        # Strategy 3: Otsu
        _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel)
        process_contours(thresh_otsu, 'Otsu')
        
        if not candidates:
            return None
            
        # Prioritize largest object (Coin > Droplet)
        best = max(candidates, key=lambda x: x['area'])
        print(f"Coin detected via {best['method']} (Score: {best['score']:.2f}, Area: {best['area']:.0f})")
        
        return np.array(best['box'])

    def get_binary_mask(self, mask):
        """
        Converts boolean mask to uint8 0/255.
        """
        return (mask * 255).astype(np.uint8)
