import sys
import os

print(f"Привет из изолированного мира!")
print(f"Версия Python: {sys.version.split()[0]}")
print(f"Текущая директория: {os.getcwd()}")
print(f"Содержимое корня (ls /): {os.listdir('/')}")