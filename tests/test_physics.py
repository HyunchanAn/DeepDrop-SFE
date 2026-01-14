import sys
import os
import numpy as np
import pytest

from deepdrop_sfe import DropletPhysics

def test_calculate_pixels_per_mm():
    # Coin radius 50px, real diameter 20mm -> 100px/20mm = 5px/mm
    scale = DropletPhysics.calculate_pixels_per_mm(50, 20)
    assert scale == 5.0
    
    # Zero case
    assert DropletPhysics.calculate_pixels_per_mm(0, 20) == 0

def test_calculate_contact_diameter():
    # Create a 100x100 mask with a 10px radius circle (Area = pi * 10^2 = 314.15)
    mask = np.zeros((100, 100), dtype=np.uint8)
    for y in range(100):
        for x in range(100):
            if (x-50)**2 + (y-50)**2 <= 10**2:
                mask[y, x] = 255
    
    # Pixels per mm = 1.0 (so diameter in pixels = diameter in mm)
    # Expected diameter = 2 * 10 = 20.0
    diameter = DropletPhysics.calculate_contact_diameter(mask, 1.0)
    assert pytest.approx(diameter, 0.1) == 20.0
    
    # Zero case
    assert DropletPhysics.calculate_contact_diameter(np.zeros((10, 10)), 1.0) == 0.0

def test_calculate_contact_angle():
    # Hypothetical case: Volume = 1 uL, Radius = 1 mm (Diameter = 2 mm)
    # V = pi*h/6 * (3r^2 + h^2)
    # For theta = 90 deg: r_contact = R_sphere, h = R_sphere
    # V = pi * R^3 * 2/3
    # If R=1mm, V = 2/3 * pi ~ 2.094 uL
    
    vol = (2.0/3.0) * np.pi
    angle = DropletPhysics.calculate_contact_angle(vol, 2.0)
    assert pytest.approx(angle, 0.1) == 90.0
    
    # Low angle case (< 90)
    # V ~ pi/4 * theta_rad * r^3 (very small theta approximation)
    # V = (pi * r^3 * theta_rad) / 4 ? 
    # Actually for 3.0 uL and 4.0 mm diameter (r=2.0)
    # Expected angle should be around 17-18 deg
    angle_low = DropletPhysics.calculate_contact_angle(3.0, 4.0)
    assert 0 < angle_low < 90

def test_calculate_owrk():
    # Mock measurements: Water (72.8), Diiodomethane (50.8)
    # If it's a PTFE-like surface, angles are ~110 and ~90
    measurements = [
        {'liquid': 'Water', 'angle': 110.0},
        {'liquid': 'Diiodomethane', 'angle': 90.0}
    ]
    total, d, p = DropletPhysics.calculate_owrk(measurements)
    assert total > 0
    assert d > 0
    # p can be near 0 for some surfaces
    
    # Not enough data
    total_none, _, _ = DropletPhysics.calculate_owrk([{'liquid': 'Water', 'angle': 100}])
    assert total_none is None

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__]))
