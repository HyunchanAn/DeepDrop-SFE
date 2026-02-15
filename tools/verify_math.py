
import numpy as np

def calculate_volume(radius_mm, contact_angle_deg):
    r = radius_mm
    theta_rad = np.radians(contact_angle_deg)
    
    # Spherical Cap Volume
    # V = (pi * r^3 / 3) * ( (1 - cos)^2 * (2 + cos) ) / sin^3
    
    sin_t = np.sin(theta_rad)
    cos_t = np.cos(theta_rad)
    
    if abs(sin_t) < 1e-9:
        return 0.0
        
    term = ((1 - cos_t)**2 * (2 + cos_t)) / (sin_t**3)
    volume_mm3 = (np.pi * r**3 / 3.0) * term
    
    return volume_mm3 # 1 mm^3 = 1 uL

r_measured = 4.94
theta_measured = 52.1
v_calculated = calculate_volume(r_measured, theta_measured)

print(f"Radius: {r_measured} mm")
print(f"Angle: {theta_measured} deg")
print(f"Calculated Volume: {v_calculated:.4f} uL")

# User claim: 58.5 uL
print("-" * 30)
print(f"User Claim (58.5 uL) -> Angle?")
# Reverse: find angle for r=4.94, V=58.5
from scipy.optimize import brentq

def volume_eq(theta_deg, target_V):
    return calculate_volume(r_measured, theta_deg) - target_V

try:
    theta_user = brentq(volume_eq, 0.1, 179.9, args=(58.5,))
    print(f"Angle for 58.5 uL: {theta_user:.2f} deg")
except:
    print("Could not find angle for 58.5 uL")

print("-" * 30)
# User claim: "100uL at 52.1 deg should be 11.8mm"
def reverse_diameter(volume_target, theta_deg):
    # V = (pi * r^3 / 3) * term
    # r^3 = 3 * V / (pi * term)
    theta_rad = np.radians(theta_deg)
    sin_t = np.sin(theta_rad)
    cos_t = np.cos(theta_rad)
    term = ((1 - cos_t)**2 * (2 + cos_t)) / (sin_t**3)
    
    r_cubed = (3 * volume_target) / (np.pi * term)
    r = r_cubed**(1/3)
    return 2 * r

d_check = reverse_diameter(100.0, 52.1)
print(f"Diameter for 100uL at 52.1 deg: {d_check:.2f} mm")
