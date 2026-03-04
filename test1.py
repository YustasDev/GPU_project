import torch
import sys
import time

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    # Тест скорости памяти (переброс тензора в память GPU)
    t0 = time.perf_counter()
    x = torch.rand(10000, 10000).cuda()
    t1 = time.perf_counter()
    print(f"Время создания и копирования: {t1 - t0:.3f} s")
    print(f"Размер тензора в GPU: {x.element_size() * x.numel() / 1e6:.1f} МБ")
    print("Tensor successfully created on GPU!")
else:
    print("!!! ALARM: WORKING ON CPU !!!")