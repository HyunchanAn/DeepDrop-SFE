import numpy as np
import cv2
from scipy.optimize import brentq

class DropletPhysics:
    """
    Physics engine for Contact Angle calculation based on Droplet Volume and Contact Diameter.
    Assumes Spherical Cap geometry.
    """
    
    # Solvent Properties (at 20C)
    LIQUID_DATA = {
        "Water": {"g": 72.8, "d": 21.8, "p": 51.0},
        "Diiodomethane": {"g": 50.8, "d": 50.8, "p": 0.0},
        "Ethylene Glycol": {"g": 48.0, "d": 29.0, "p": 19.0},
        "Glycerol": {"g": 64.0, "d": 34.0, "p": 30.0},
        "Formamide": {"g": 58.0, "d": 39.0, "p": 19.0}
    }

    @staticmethod
    def calculate_pixels_per_mm(coin_radius_pixel, real_coin_diameter_mm):
        """
        Calculates scale factor (pixels per mm).
        Note: Input is coin RADIUS in pixels (from perspective correction), 
        and real DIAMETER in mm.
        """
        # Coin Diameter in pixels = 2 * Radius
        coin_diameter_pixel = 2 * coin_radius_pixel
        if coin_diameter_pixel == 0:
            return 0
        return coin_diameter_pixel / real_coin_diameter_mm

    @staticmethod
    def calculate_contact_diameter(droplet_mask, pixels_per_mm):
        """
        Calculates the real contact diameter of the droplet from its mask.
        Assumes the mask represents the circular base (top-view).
        """
        # Calculate Area in pixels
        area_pixels = np.sum(droplet_mask > 0)
        
        if area_pixels == 0:
            return 0.0
            
        # Equivalent Diameter in pixels (assuming circle)
        # Area = pi * (d/2)^2 => d = 2 * sqrt(Area / pi)
        diameter_pixels = 2 * np.sqrt(area_pixels / np.pi)
        
        # Convert to mm
        diameter_mm = diameter_pixels / pixels_per_mm
        return diameter_mm

    @staticmethod
    def calculate_contact_angle(volume_ul, diameter_mm):
        """
        Calculates Contact Angle (Theta) given Droplet Volume (uL) and Contact Diameter (mm).
        Uses numerical inversion of the Spherical Cap Volume formula.
        
        Formula:
        V = (pi * r^3 * (1 - cos(theta))^2 * (2 + cos(theta))) / (3 * sin(theta)^3)
        where r = diameter / 2
        """
        if diameter_mm <= 0 or volume_ul <= 0:
            return 0.0
            
        r = diameter_mm / 2.0
        target_V = volume_ul # 1 uL = 1 mm^3
        
        # Function to find root for: f(theta) - target_V = 0
        def volume_eq(theta_deg):
            if theta_deg <= 0.1 or theta_deg >= 179.9:
                return -target_V # Avoid singularities, assume V=Large for theta=180
                
            theta_rad = np.radians(theta_deg)
            sin_t = np.sin(theta_rad)
            cos_t = np.cos(theta_rad)
            
            # Spherical Cap Volume Factor
            # term = (1-cos)(1-cos)(2+cos) / sin^3
            term = ((1 - cos_t)**2 * (2 + cos_t)) / (sin_t**3)
            
            V_calc = (np.pi * r**3 / 3.0) * term
            return V_calc - target_V
            
        # Numerical Solve
        # Theta range: 0.1 to 179.9 degrees
        try:
            # Check signs at endpoints to ensure root exists
            v_low = volume_eq(0.1)   # Should be negative (calculated V near 0) - target_V
            v_high = volume_eq(179.9) # Should be positive (calculated V near inf) - target_V
            
            if v_low * v_high > 0:
                # Should not happen for valid V, r
                print(f"Physics Warning: Root finding failed. v_low={v_low}, v_high={v_high}")
                return 0.0
                
            theta_sol = brentq(volume_eq, 0.1, 179.9)
            return theta_sol
        except Exception as e:
            print(f"Physics Error: {e}")
            return 0.0

    @staticmethod
    def calculate_owrk(measurements):
        """
        OWRK Surface Energy Calculation.
        measurements: list of dict {'liquid': str, 'angle': float}
        """
        X_points = []
        Y_points = []
        
        for m in measurements:
            name = m['liquid']
            angle = m['angle']
            
            if name not in DropletPhysics.LIQUID_DATA:
                continue
                
            props = DropletPhysics.LIQUID_DATA[name]
            
            if props['d'] <= 0:
                continue
                
            theta_rad = np.radians(angle)
            
            # Y = (gamma_L * (1 + cos_theta)) / (2 * sqrt(gamma_L_d))
            y_val = (props['g'] * (1 + np.cos(theta_rad))) / (2 * np.sqrt(props['d']))
            
            # X = sqrt(gamma_L_p / gamma_L_d)
            x_val = np.sqrt(props['p'] / props['d'])
            
            X_points.append(x_val)
            Y_points.append(y_val)
            
        if len(X_points) < 2:
            return None, 0.0, 0.0
            
        A = np.vstack([X_points, np.ones(len(X_points))]).T
        slope, intercept = np.linalg.lstsq(A, Y_points, rcond=None)[0]
        
        gamma_s_p = slope**2
        gamma_s_d = intercept**2
        total_sfe = gamma_s_d + gamma_s_p
        
        return total_sfe, gamma_s_d, gamma_s_p
