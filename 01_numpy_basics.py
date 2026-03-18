import numpy as np
import cv2

if __name__ == "__main__":

    # 1. Создаем "Черный квадрат" Малевича
    # Формат: (Высота, Ширина, Каналы). Тип данных: uint8 (0..255)
    height, width = 400, 600
    black_canvas = np.zeros((height, width, 3), dtype=np.uint8)

    print(f"Shape: {black_canvas.shape}")  # (400, 600, 3)
    print(f"Data Type: {black_canvas.dtype}") # uint8

    # 2. Красим пиксели (Slicing)
    # Не забываем, что в OpenCV порядок цветов BGR (Blue, Green, Red), а не RGB
    # Закрасим левую половину в Синий
    black_canvas[:, :300] = (255, 0, 0) # (B=255, G=0, R=0)

    # Закрасим правую половину в Зеленый
    black_canvas[:, 300:] = (0, 255, 0)

    # 3. Вырезаем кусок (ROI - Region of Interest)
    # Берем квадрат по центру
    center_y, center_x = height // 2, width // 2
    roi = black_canvas[center_y-50:center_y+50, center_x-50:center_x+50]

    # Изменим цвет этого куска на Красный (прямо в оригинальной картинке)
    roi[:] = (0, 0, 255)

    # 4. Сохраняем результат (если мы в WSL2 и не можем показать окно)
    cv2.imwrite("numpy_art.jpg", black_canvas)
    print("Image saved to numpy_art.jpg. Open it from Windows Explorer!")