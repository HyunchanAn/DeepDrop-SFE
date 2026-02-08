import sys
import os
import platform
import subprocess

def get_cpu_info():
    try:
        if platform.system() == "Windows":
            cmd = "powershell -Command \"Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors | ConvertTo-Json\""
            result = subprocess.check_output(cmd, shell=True).decode()
            return result
    except:
        return "CPU 정보를 가져올 수 없습니다."

def check_torch():
    try:
        import torch
        info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        }
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
        return info
    except ImportError:
        return "PyTorch가 설치되어 있지 않습니다."

def main():
    print("="*50)
    print("시스템 및 하드웨어 가속 진단 보고서")
    print("="*50)
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Python: {sys.version}")
    print(f"CPU 정보: {get_cpu_info()}")
    
    print("-"*50)
    print("PyTorch 및 GPU 가속 확인:")
    torch_info = check_torch()
    if isinstance(torch_info, dict):
        for k, v in torch_info.items():
            print(f"  - {k}: {v}")
    else:
        print(f"  {torch_info}")
    
    print("-"*50)
    print("NVIDIA-SMI 상태:")
    try:
        os.system("nvidia-smi")
    except:
        print("  nvidia-smi를 실행할 수 없습니다.")
    print("="*50)

if __name__ == "__main__":
    main()
