import numpy as np
from deepdrop_sfe import DropletPhysics

def main():
    print("=== DeepDrop-SFE Core Library Example ===")
    
    # Example 1: Calculate Contact Angle from Volume and Diameter
    vol = 3.0 # uL
    diam = 9.13 # mm
    
    print(f"\n[Input] Volume: {vol} uL, Diameter: {diam} mm")
    
    # Basic Calculation
    # angle = DropletPhysics.calculate_contact_angle(vol, diam)
    # print(f"Calculated Angle: {angle:.2f} degrees")
    
    # Calculation with Diagnostics
    angle, diag = DropletPhysics.calculate_contact_angle(vol, diam, return_info=True)
    print(f"Calculated Angle (Enhanced): {angle:.2f} degrees")
    print(f"Diagnostics: status={diag['status']}, v_low={diag['v_low']:.2f}")

    # Example 2: SFE Calculation (OWRK)
    print("\n--- SFE Calculation ---")
    data = [
        {'liquid': 'Water', 'angle': 110.0},        # High angle (Hydrophobic)
        {'liquid': 'Diiodomethane', 'angle': 65.0}  # Lower angle
    ]
    sfe, d, p = DropletPhysics.calculate_owrk(data)
    
    if sfe:
        print(f"Total SFE: {sfe:.2f} mN/m")
        print(f" - Dispersive: {d:.2f} mN/m")
        print(f" - Polar:      {p:.2f} mN/m")
    else:
        print("SFE Calculation failed (insufficient data)")

if __name__ == "__main__":
    main()
