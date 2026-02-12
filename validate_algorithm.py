import cv2
import numpy as np
import os
import torch
from deepdrop_sfe import AIContactAngleAnalyzer, DropletPhysics, PerspectiveCorrector

def run_validation():
    print("="*50)
    print("DeepDrop-AnyView Algorithm Validation")
    print("="*50)
    
    # 1. Setup Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    analyzer = AIContactAngleAnalyzer(device=device)
    corrector = PerspectiveCorrector()
    
    # 2. Benchmark Cases
    # Case 1: metal_water.png (Log reference: 3.0uL -> ~2.30 deg)
    test_cases = [
        {
            "name": "Metal Water (3.0uL)",
            "image_path": "metal_water.png",
            "volume_ul": 3.0,
            "expected_angle": 2.30,
            "ref_diameter_mm": 24.0 # 100 KRW
        },
        {
             "name": "Metal Glycerin (3.0uL)",
             "image_path": "metal_glycerin.png",
             "volume_ul": 3.0,
             "ref_diameter_mm": 24.0
        }
    ]
    
    for case in test_cases:
        print(f"\n[Testing Case: {case['name']}]")
        img_path = case['image_path']
        if not os.path.exists(img_path):
            print(f"Skipping: {img_path} not found.")
            continue
            
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # A. Detect Coin
        coin_box = analyzer.auto_detect_coin_candidate(image)
        if coin_box is None:
            print("Fail: Could not detect coin.")
            continue
            
        analyzer.set_image(image_rgb)
        coin_mask, _ = analyzer.predict_mask(box=coin_box)
        coin_mask_binary = analyzer.get_binary_mask(coin_mask)
        
        # B. Homography
        H, warped_size, coin_info, _ = corrector.find_homography(image_rgb, coin_mask_binary)
        if H is None:
            print("Fail: Homography failed.")
            continue
            
        warped_img = corrector.warp_image(image_rgb, H, warped_size)
        
        # C. Detect Droplet on Warped Image
        analyzer.set_image(warped_img)
        droplet_box = analyzer.auto_detect_droplet_candidate(warped_img)
        if droplet_box is None:
             print("Fail: Could not detect droplet.")
             continue
             
        droplet_mask, _ = analyzer.predict_mask(box=droplet_box)
        
        # D. Physics Calculation
        _, _, radius_px = coin_info
        pixels_per_mm = DropletPhysics.calculate_pixels_per_mm(radius_px, case['ref_diameter_mm'])
        
        # Use enhanced fitting method
        contact_diameter_mm = DropletPhysics.calculate_contact_diameter(droplet_mask, pixels_per_mm, method="fitting")
        contact_angle, diag = DropletPhysics.calculate_contact_angle(case['volume_ul'], contact_diameter_mm, return_info=True)
        
        # Circularity Score (Reliability)
        contours, _ = cv2.findContours(droplet_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            _, radius = cv2.minEnclosingCircle(cnt)
            circle_area = np.pi * (radius**2)
            circularity = area / circle_area if circle_area > 0 else 0
        else:
            circularity = 0
            
        print(f"  - Contact Diameter: {contact_diameter_mm:.3f} mm")
        print(f"  - Calculated Angle: {contact_angle:.3f} deg")
        if "expected_angle" in case:
            diff = contact_angle - case['expected_angle']
            print(f"  - Target Angle: {case['expected_angle']} deg (Diff: {diff:+.3f})")
            
        print(f"  - Reliability Indices:")
        print(f"    * Circularity Score: {circularity:.4f}")
        print(f"    * Sensitivity (dV): {diag.get('sensitivity_v', 0):.4f} deg/%")
        print(f"    * Sensitivity (dD): {diag.get('sensitivity_d', 0):.4f} deg/%")
        print(f"  - Solver Status: {diag['status']}")

if __name__ == "__main__":
    run_validation()
