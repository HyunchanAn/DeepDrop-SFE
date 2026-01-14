import sys
import os
import numpy as np
import pytest

from deepdrop_sfe import DropletPhysics

def test_90_degree_diagnostics():
    # Case: V = 2/3 * pi * r^3
    # For r = 1.0 (D = 2.0), V = 2.094395
    vol = (2.0/3.0) * np.pi
    angle, diag = DropletPhysics.calculate_contact_angle(vol, 2.0, return_info=True)
    
    print(f"\nDiagnostic Info for 90deg case:")
    for k, v in diag.items():
        print(f"  {k}: {v}")
    
    assert diag["status"] == "Success"
    assert pytest.approx(angle, 0.1) == 90.0
    assert diag["v_low"] < 0
    assert diag["v_high"] > 0

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__]))
