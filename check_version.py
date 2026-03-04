import sys
import torch

def check_versions():
    print("--- Информация о системе ---")
    
    # Python Version
    python_version = sys.version
    print(f"Версия Python: {python_version}")
    
    # PyTorch Version
    pytorch_version = torch.__version__
    print(f"Версия PyTorch: {pytorch_version}")
    
    # CUDA Availability and Version
    cuda_available = torch.cuda.is_available()
    print(f"Доступность GPU (CUDA): {'Доступен' if cuda_available else 'Недоступен'}")
    
    if cuda_available:
        cuda_version = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Версия CUDA: {cuda_version}")
        print(f"Используемый GPU: {gpu_name}")
    else:
        print("CUDA не обнаружена или GPU недоступен.")

if __name__ == "__main__":
    check_versions()
