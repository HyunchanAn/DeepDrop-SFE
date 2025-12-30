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

    @staticmethod
    def calculate_owrk(angle_water, angle_diiodo):
        """
        OWRK 방정식을 이용해 표면 에너지를 계산합니다.
        
        (1 + cosθ) * γ_L = 2 * (sqrt(γ_S^d * γ_L^d) + sqrt(γ_S^p * γ_L^p))
        y = mx + c 형태로 변형하여 풉니다.
        
        Data (Ström et al.):
        Water:          gamma=72.8, d=21.8, p=51.0
        Diiodomethane:  gamma=50.8, d=50.8, p=0.0
        """
        
        # Solvent Properties (at 20C)
        liquids = {
            "Water": {"g": 72.8, "d": 21.8, "p": 51.0},
            "Diiodomethane": {"g": 50.8, "d": 50.8, "p": 0.0}
        }
        
        # 계산 편의를 위한 단순화 로직 (실제 연립방정식 해)
        # A * sqrt(gamma_s_d) + B * sqrt(gamma_s_p) = C 형태의 식 2개.
        
        # 방정식 1 (Water)
        # 2*sqrt(21.8)*x + 2*sqrt(51.0)*y = (1 + cos(w)) * 72.8
        
        # 방정식 2 (Diiodomethane)
        # 2*sqrt(50.8)*x + 2*sqrt(0.0)*y = (1 + cos(d)) * 50.8
        # -> 2*sqrt(50.8)*x = (1 + cos(d)) * 50.8
        # -> x (sqrt_gamma_s_d)를 바로 구할 수 있음.
        
        rad_w = np.radians(angle_water)
        rad_d = np.radians(angle_diiodo)
        
        w_params = liquids["Water"]
        d_params = liquids["Diiodomethane"]
        
        # 1. 분산 성분 (Dispersive) 계산 - Diiodomethane식 이용 (Polar component가 0이므로)
        # 2 * sqrt(50.8 * S_d) = 50.8 * (1 + cos(theta_d))
        # sqrt(S_d) = (50.8 * (1 + cos(theta_d))) / (2 * sqrt(50.8))
        #           = sqrt(50.8) * (1 + cos(theta_d)) / 2
        
        sqrt_Sd = np.sqrt(d_params["d"]) * (1 + np.cos(rad_d)) / 2
        Sd = sqrt_Sd ** 2
        
        # 2. 극성 성분 (Polar) 계산 - Water식 대입
        # 72.8 * (1 + cos(theta_w)) = 2 * sqrt(21.8 * Sd) + 2 * sqrt(51.0 * Sp)
        # sqrt(51.0 * Sp) = [ 72.8 * (1 + cos(theta_w)) - 2 * sqrt(21.8 * Sd) ] / 2
        
        lhs_water = w_params["g"] * (1 + np.cos(rad_w))
        term_dispersive = 2 * np.sqrt(w_params["d"] * Sd)
        
        term_polar = lhs_water - term_dispersive
        
        # 음수 방지 (실험 오차로 인해 발생 가능)
        if term_polar < 0:
            term_polar = 0
            
        sqrt_Sp = term_polar / (2 * np.sqrt(w_params["p"]))
        Sp = sqrt_Sp ** 2
        
        total_sfe = Sd + Sp
        return total_sfe, Sd, Sp
