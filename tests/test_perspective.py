import sys
import os
import numpy as np
import pytest
import cv2

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from perspective import PerspectiveCorrector

def test_find_homography_basic():
    corrector = PerspectiveCorrector()
    
    # Create a mock image and an elliptical mask
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    mask = np.zeros((500, 500), dtype=np.uint8)
    
    # Draw an ellipse (representing an oblique coin)
    # Center (250, 250), Axes (100, 50), Angle 30
    cv2.ellipse(mask, (250, 250), (100, 50), 30, 0, 360, 255, -1)
    
    H, warped_size, coin_info, fitted_ellipse = corrector.find_homography(image, mask)
    
    assert H is not None
    assert H.shape == (3, 3)
    assert warped_size == (500, 500)
    
    # Check coin_info: radius should be around the major axis (100)
    cx, cy, radius = coin_info
    assert pytest.approx(cx, 5) == 250
    assert pytest.approx(cy, 5) == 250
    assert pytest.approx(radius, 5) == 100

def test_warp_point():
    corrector = PerspectiveCorrector()
    # Simple identity homography
    H = np.eye(3, dtype=np.float32)
    
    point = (100, 200)
    warped = corrector.warp_point(point, H)
    
    assert pytest.approx(warped[0]) == 100
    assert pytest.approx(warped[1]) == 200

def test_empty_mask():
    corrector = PerspectiveCorrector()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    
    H, _, _, _ = corrector.find_homography(image, mask)
    assert H is None
