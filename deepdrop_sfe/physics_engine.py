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
    def calculate_contact_diameter(droplet_mask, pixels_per_mm, method="fitting", return_extra=False):
        """
        Calculates the real contact diameter of the droplet from its mask.
        
        Methods:
        - 'area': Equivalent diameter based on total pixel area.
        - 'fitting': Geometric diameter using minimum enclosing circle (robust to noise).
        """
        if pixels_per_mm <= 0:
            if return_extra: return 0.0, 0.0
            return 0.0

        # Find contours
        contours, _ = cv2.findContours(droplet_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            if return_extra: return 0.0, 0.0
            return 0.0
        
        # Select the largest contour
        cnt = max(contours, key=cv2.contourArea)
        area_pixels = cv2.contourArea(cnt)
        
        if area_pixels < 10:
            if return_extra: return 0.0, 0.0
            return 0.0
            
        # Calculate Circularity (Reliability)
        # Circularity = Area / (pi * r_enclosing^2)
        (cx, cy), radius_enclosing = cv2.minEnclosingCircle(cnt)
        circle_area = np.pi * (radius_enclosing**2)
        circularity = area_pixels / circle_area if circle_area > 0 else 0

        if method == "area":
            # Area = pi * (d/2)^2 => d = 2 * sqrt(Area / pi)
            diameter_pixels = 2 * np.sqrt(area_pixels / np.pi)
        else:
            # Geometric fitting
            diameter_pixels = 2 * radius_enclosing
            
        # Convert to mm
        diameter_mm = diameter_pixels / pixels_per_mm
        
        if return_extra:
            return diameter_mm, circularity
        return diameter_mm

    @staticmethod
    def calculate_contact_angle(volume_ul, diameter_mm, return_info=False):
        """
        Calculates Contact Angle (Theta) given Droplet Volume (uL) and Contact Diameter (mm).
        Uses numerical inversion of the Spherical Cap Volume formula.
        
        Formula:
        V = (pi * r^3 * (1 - cos(theta))^2 * (2 + cos(theta))) / (3 * sin(theta)^3)
        where r = diameter / 2
        """
        diag = {
            "v_low": 0.0, "v_high": 0.0, "target_V": volume_ul, 
            "r": diameter_mm / 2.0 if diameter_mm else 0.0, 
            "v_full": 0.0, "status": "Initializing"
        }

        if not np.isfinite(volume_ul) or not np.isfinite(diameter_mm):
            diag["status"] = "Error: Non-finite inputs"
            if return_info: return 0.0, diag
            return 0.0

        if diameter_mm <= 0 or volume_ul <= 0:
            diag["status"] = "Error: Zero or negative inputs"
            if return_info: return 0.0, diag
            return 0.0
            
        r = diameter_mm / 2.0
        target_V = volume_ul # 1 uL = 1 mm^3
        v_full = (4.0/3.0) * np.pi * (r**3)
        diag["v_full"] = v_full
        
        # Function to find root for: f(theta) - target_V = 0
        def volume_eq(theta_deg):
            # Clip to avoid division by zero and extreme values
            theta_deg_clipped = np.clip(theta_deg, 1e-8, 179.9999)
            theta_rad = np.radians(theta_deg_clipped)
            
            # Using stable form to avoid sin(theta)^3 in denominator when theta is small
            # V = (pi * r^3 / 3) * ( (1-cos(theta))^2 * (2+cos(theta)) / sin(theta)^3 )
            
            sin_t = np.sin(theta_rad)
            cos_t = np.cos(theta_rad)
            
            if theta_deg_clipped < 0.1:
                # Small angle approximation for numerical stability
                # Using Taylor expansion: V approx (pi * r^3 * theta_rad) / 4
                term = (3.0 * theta_rad) / 4.0
            else:
                term = ((1 - cos_t)**2 * (2 + cos_t)) / (sin_t**3 + 1e-18)
                
            V_calc = (np.pi * r**3 / 3.0) * term
            return V_calc - target_V
            
        try:
            # Check boundaries for brentq
            v_low_val = volume_eq(1e-8)
            v_high_val = volume_eq(179.99)
            diag["v_low"] = v_low_val
            diag["v_high"] = v_high_val
            
            if v_low_val > 0:
                diag["status"] = "Capped: Near-zero volume (Minimum reached)"
                if return_info: return 0.0, diag
                return 0.0
                
            if v_high_val < 0:
                diag["status"] = "Capped: Volume exceeds sphere (Maximum reached)"
                if return_info: return 180.0, diag
                return 180.0

            # Solve using brentq with safety margins and explicit tolerance
            theta_sol = brentq(volume_eq, 1e-8, 179.99, xtol=1e-6)
            diag["status"] = "Success"
            
            # --- Reliability Metrics ---
            # 1. Sensitivity Analysis (Numerical Gradient)
            eps_v = target_V * 0.01
            eps_d = diameter_mm * 0.01
            
            def get_angle(v, d):
                r_local = d / 2.0
                def eq(t):
                    tr = np.radians(np.clip(t, 1e-7, 179.99))
                    val = (np.pi * r_local**3 / 3.0) * ((1 - np.cos(tr))**2 * (2 + np.cos(tr))) / (np.sin(tr)**3)
                    return val - v
                try: return brentq(eq, 1e-7, 179.9)
                except: return theta_sol

            angle_v_plus = get_angle(target_V + eps_v, diameter_mm)
            diag["sensitivity_v"] = (angle_v_plus - theta_sol) / 1.0 # % change in V -> change in Angle
            
            angle_d_plus = get_angle(target_V, diameter_mm + eps_d)
            diag["sensitivity_d"] = (angle_d_plus - theta_sol) / 1.0 # % change in D -> change in Angle

            if return_info: return theta_sol, diag
            return theta_sol
        except Exception as e:
            diag["status"] = f"Error: {e}"
            if return_info: return 0.0, diag
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
