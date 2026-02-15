
import numpy as np

def calculate_volume(diameter_mm, theta_deg):
    """
    Spherical Cap Volume Formula:
    R = D / (2 * sin(theta))
    V = (pi * R^3 / 3) * (2 - 3cos(theta) + cos^3(theta))
    """
    theta_rad = np.radians(theta_deg)
    R = (diameter_mm / 2) / np.sin(theta_rad)
    
    # Volume in mm^3 (which equals micro-liter, uL)
    vol = (np.pi * R**3 / 3) * (2 - 3*np.cos(theta_rad) + np.cos(theta_rad)**3)
    return vol

diameter = 8.84  # From previous analysis result

print(f"--- Volume Estimation for Diameter = {diameter} mm ---")
for angle in [30, 45, 60, 90, 100]:
    v = calculate_volume(diameter, angle)
    print(f"Angle: {angle:3d}° -> Volume: {v:6.2f} µL")
