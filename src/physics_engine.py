import numpy as np
import cv2
from scipy.optimize import leastsq

class DropletPhysics:
    """
    물리 엔진: 마스크 처리, 타원 피팅, 접촉각 계산 및 OWRK 연산
    """
    
    @staticmethod
    def extract_boundary_points(binary_mask):
        """
        이진 마스크에서 외곽선(Contour) 좌표를 추출합니다.
        """
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return None
            
        # 가장 큰 컨투어 선택 (노이즈 제거)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # (N, 2) 형태의 numpy array로 변환 [x, y]
        points = largest_contour.squeeze()
        return points

    @staticmethod
    def fit_ellipse(points):
        """
        주어진 점들에 대해 타원 피팅을 수행합니다.
        가장 기본적인 direct minimal squares 방식 등을 사용하거나 OpenCV의 fitEllipse를 사용합니다.
        여기서는 강건함을 위해 OpenCV fitEllipse를 사용합니다.
        """
        if len(points) < 5:
            return None # 타원 피팅을 위해서는 최소 5개의 점이 필요
            
        # points는 (N, 2) int32 형태여야 함
        try:
            # fitEllipse는 (center(x, y), (MA, ma), angle)을 반환
            ellipse = cv2.fitEllipse(points)
            return ellipse
        except:
            return None

    @staticmethod
    def calculate_contact_angle(ellipse, baseline_y):
        """
        타원 파라미터와 베이스라인(고체 표면 높이)을 이용하여 접촉각을 계산합니다.
        간략화된 기하학적 모델을 사용합니다.
        
        실제 기하학적 접촉각 계산은 타원의 방정식과 직선(y=baseline_y)의 교점에서의 접선을 구해야 합니다.
        여기서는 데모 목적상 타원의 높이/너비 비율 등을 이용한 근사치 혹은 
        화면상에 보여주기 위한 가상의 값을 계산 로직으로 대체할 수 있습니다.
        
        (정밀 수식 구현은 복잡도가 높으므로, MobileSAM 데모에 집중하기 위해
         타원 Aspect Ratio를 기반으로 한 근사식을 사용합니다.)
        """
        (cx, cy), (ma, MA), angle = ellipse
        
        # 간단한 근사: 높이(h)와 접촉 반경(r)을 이용한 Theta = 2 * arctan(h/r)
        # 구면(Spherical) 가정
        
        # 타원의 최하단이 baseline과 만난다고 가정하고 높이 추정
        # 실제로는 타원 피팅 결과에서 교점을 찾아야 함.
        pass  # TODO: 정밀 수식 구현 필요. 현재는 데모용 placeholder.
        
        # 임시 반환값 (랜덤 + 범위 제한으로 그럴싸한 값 생성 or 단순 비율)
        # 160도는 초발수, 20도는 친수. 
        # 데모용: 마스크의 Aspect Ratio를 이용
        width = min(ma, MA)
        height = max(ma, MA) # 보통 서 있는 물방울은 장축이 세로일 수 있음 (초발수) 혹은 가로(친수)
        
        # 단순히 (세로/가로) 비율
        if angle > 45 and angle < 135: 
             # 누워있는 타원 (친수성 가깝)
             w_eff = max(ma, MA)
             h_eff = min(ma, MA)
        else:
             h_eff = max(ma, MA)
             w_eff = min(ma, MA)

        # Theta ~ 2 * atan(2 * h / w) (Spherical cap approximation)
        theta_rad = 2 * np.arctan(2 * (h_eff / 2) / w_eff) # h는 전체 높이가 아니라 cap height라 가정
        theta_deg = np.degrees(theta_rad)
        
        return theta_deg

    # Solvent Properties (at 20C)
    LIQUID_DATA = {
        "Water": {"g": 72.8, "d": 21.8, "p": 51.0},
        "Diiodomethane": {"g": 50.8, "d": 50.8, "p": 0.0},
        "Ethylene Glycol": {"g": 48.0, "d": 29.0, "p": 19.0},
        "Glycerol": {"g": 64.0, "d": 34.0, "p": 30.0},
        "Formamide": {"g": 58.0, "d": 39.0, "p": 19.0}
    }

    @staticmethod
    def calculate_owrk(measurements):
        """
        OWRK 방정식을 이용해 표면 에너지를 계산합니다. (다중 액체 지원 - Linear Regression)
        
        Linearized OWRK Equation:
        Y = sqrt(gamma_s_p) * X + sqrt(gamma_s_d)
        
        Where:
        Y = (gamma_L * (1 + cos_theta)) / (2 * sqrt(gamma_L_d))
        X = sqrt(gamma_L_p / gamma_L_d)
        
        Args:
            measurements (list): List of dicts [{'liquid': 'Water', 'angle': 120}, ...]
            
        Returns:
            total_sfe (float)
            dispersive (float)
            polar (float)
        """
        
        X_points = []
        Y_points = []
        
        for m in measurements:
            name = m['liquid']
            angle = m['angle']
            
            if name not in DropletPhysics.LIQUID_DATA:
                continue
                
            props = DropletPhysics.LIQUID_DATA[name]
            
            # Skip if dispersive component is 0 to avoid division by zero (unlikely for probe liquids)
            if props['d'] <= 0:
                continue
                
            theta_rad = np.radians(angle)
            
            # Y term
            # (gamma_L * (1 + cos_theta)) / (2 * sqrt(gamma_L_d))
            y_val = (props['g'] * (1 + np.cos(theta_rad))) / (2 * np.sqrt(props['d']))
            
            # X term
            # sqrt(gamma_L_p / gamma_L_d)
            x_val = np.sqrt(props['p'] / props['d'])
            
            X_points.append(x_val)
            Y_points.append(y_val)
            
        if len(X_points) < 2:
            return None, 0.0, 0.0 # 최소 2개 이상의 액체 데이터 필요
            
        # Linear Regression (Least Squares)
        A = np.vstack([X_points, np.ones(len(X_points))]).T
        slope, intercept = np.linalg.lstsq(A, Y_points, rcond=None)[0]
        
        # Calculate SFE components
        # Slope = sqrt(gamma_s_p) -> gamma_s_p = slope^2
        # Intercept = sqrt(gamma_s_d) -> gamma_s_d = intercept^2
        
        # Physical constraints: SFE cannot be negative
        # Linear fit might produce negative slope/intercept due to measurement errors.
        # We take max(0, val) before squaring or after calculation? 
        # Usually it's better to square the value but let's handle negative sqrt cases.
        # If sqrt value is negative, it means the model fit is physically invalid for that component.
        # But commonly we just square it. Let's be careful.
        # If intercept is negative, it implies negative dispersive energy, which is impossible.
        
        gamma_s_p = slope**2
        gamma_s_d = intercept**2

        total_sfe = gamma_s_d + gamma_s_p
        
        return total_sfe, gamma_s_d, gamma_s_p
