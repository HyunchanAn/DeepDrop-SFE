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
        Improved to handle multi-mask output and filter background.
        """
        if points is None and box is None:
             # Default: Center point strategy
             h, w = self.predictor.original_size
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
            
        # Pick the best mask.
        # Strategy: Droplets should be small. Background mask is large.
        # If a box is provided, we prefer the mask that is concentrated within the box.
        
        valid_masks = []
        img_area = self.predictor.original_size[0] * self.predictor.original_size[1]
        
        for i in range(len(masks)):
            mask = masks[i]
            mask_area = np.sum(mask)
            if mask_area < 10 or mask_area > 0.8 * img_area:
                continue
            
            # Boundary check: droplets should not touch the image edge in most cases
            # after perspective correction (they should be in the center)
            edge_pixels = np.sum(mask[0, :]) + np.sum(mask[-1, :]) + np.sum(mask[:, 0]) + np.sum(mask[:, -1])
            edge_ratio = edge_pixels / (2 * (mask.shape[0] + mask.shape[1]))
            
            # Score can be a combination of SAM score and local containment
            final_score = scores[i]
            
            # Penalty for touching edges (likely background leak)
            if edge_ratio > 0.05:
                final_score *= 0.1
            
            # If box is provided, calculate IoU or containment ratio
            if box is not None:
                x1, y1, x2, y2 = box
                # Extract mask region within box
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
            # Fallback to the highest score
            idx = np.argmax(scores)
            return masks[idx], scores[idx]
            
        # Pick the one with highest final_score
        best = max(valid_masks, key=lambda x: x['final_score'])
        return best['mask'], best['sam_score']

    def auto_detect_coin_candidate(self, image_cv2):
        """
        Uses Hull Circularity and Solidity to find oblique coin candidates.
        """
        gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        rows = gray.shape[0]
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        candidates = []

        # Strategy 1: Hough Circle
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=rows/10,
                                   param1=100, param2=25, minRadius=20, maxRadius=rows//2)
        if circles is not None:
             # Do NOT cast to uint8 directly here to avoid overflow!
             circles = circles[0, :]
             # Pick the largest circle by radius
             best_circle = max(circles, key=lambda x: x[2])
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

    def auto_detect_droplet_candidate(self, image_cv2, exclude_box=None):
        """
        Uses basic CV to find droplet candidates.
        exclude_box: [x1, y1, x2, y2] to ignore (e.g. coin area)
        """
        gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Look for small circular-ish things
        # Droplets often have high local contrast
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
            
            # Check if this box overlaps with exclude_box (coin)
            if exclude_box is not None:
                ex1, ey1, ex2, ey2 = exclude_box
                # Intersection
                ix1 = max(x, ex1)
                iy1 = max(y, ey1)
                ix2 = min(x+w, ex2)
                iy2 = min(y+h, ey2)
                if ix1 < ix2 and iy1 < iy2:
                    intersect_area = (ix2 - ix1) * (iy2 - iy1)
                    if intersect_area > 0.5 * area:
                        continue
            
            # Metric: Solidity and Circularity
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0: continue
            solidity = float(area) / hull_area
            
            peri = cv2.arcLength(cnt, True)
            if peri == 0: continue
            circularity = 4 * np.pi * area / (peri * peri)
            
            # Droplets are usually convex and somewhat circular
            if solidity > 0.8 and circularity > 0.5:
                candidates.append({
                    'box': [max(0, x-5), max(0, y-5), min(image_cv2.shape[1], x+w+5), min(image_cv2.shape[0], y+h+5)],
                    'area': area,
                    'score': solidity * circularity
                })
        
        if not candidates:
            return None
            
        # Pick the best candidate (highest solidity*circularity)
        best = max(candidates, key=lambda x: x['score'])
        return np.array(best['box'])

    def get_binary_mask(self, mask):
        """
        Converts boolean mask to uint8 0/255.
        """
        return (mask * 255).astype(np.uint8)
