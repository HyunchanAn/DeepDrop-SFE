
import cv2
import sys
import os
import numpy as np
import glob

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deepdrop_sfe.ai_engine import AIContactAngleAnalyzer
from deepdrop_sfe.perspective import PerspectiveCorrector

def test_oblique_images():
    # Load model (Mocking or Real? Real is better for visual check)
    try:
        # Use CPU for quick test or CUDA if available. 
        # The user has RTX 5080, so CUDA is fine.
        analyzer = AIContactAngleAnalyzer()
        corrector = PerspectiveCorrector()
    except Exception as e:
        print(f"Model load failed: {e}")
        return

    # Target images
    image_paths = glob.glob("metal_*.png")
    if not image_paths:
        print("No metal_*.png images found in current directory.")
        return

    scale_factor = 1.0 # Process at full resolution or scale down? 
    # Full resolution is better for detail.

    for img_path in image_paths:
        print(f"\nProcessing {img_path}...")
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load {img_path}")
            continue
            
        # 1. Auto Detect Coin
        print("  1. Detecting Coin...")
        coin_box = analyzer.auto_detect_coin_candidate(image)
        
        if coin_box is None:
            print("  [FAILED] Coin detection failed.")
            continue
            
        print(f"  [SUCCESS] Coin Box: {coin_box}")
        
        # Visualize Box
        vis_img = image.copy()
        x1, y1, x2, y2 = map(int, coin_box)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # 2. Predict Mask (for Homography)
        print("  2. Generating Mask...")
        analyzer.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        coin_mask, score = analyzer.predict_mask(box=coin_box)
        
        mask_binary = analyzer.get_binary_mask(coin_mask)
        
        # Visualize Mask
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_img, contours, -1, (255, 0, 0), 2)
        
        # 3. Homography
        print("  3. Calculating Homography...")
        H, size, info, ellipse = corrector.find_homography(image, mask_binary)
        
        if H is None:
            print("  [FAILED] Homography calculation failed.")
            cv2.imwrite(f"{img_path}_failed.jpg", vis_img)
            continue
            
        print("  [SUCCESS] Homography computed.")
        
        # Visualize Ellipse
        (ex, ey), (eda, edb), eangle = ellipse
        cv2.ellipse(vis_img, ((ex, ey), (eda, edb), eangle), (0, 0, 255), 2)
        
        # Save Debug Image
        cv2.imwrite(f"{img_path}_debug.jpg", vis_img)
        
        # Warp Image
        warped = corrector.warp_image(image, H, size)
        cv2.imwrite(f"{img_path}_warped.jpg", warped)
        print(f"  Saved debug images: {img_path}_debug.jpg, {img_path}_warped.jpg")

if __name__ == "__main__":
    test_oblique_images()
