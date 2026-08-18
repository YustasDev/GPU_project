from ultralytics import YOLO
import torch
import cv2

if __name__ == "__main__":

    print(f"PyTorch видит GPU: {torch.cuda.is_available()}")

    # 1. Загрузка модели
    # При первом запуске библиотека сама скачает файл yolo11m.pt (~40 МБ) с серверов
    print("Загружаем модель YOLO11m...")
    model = YOLO("yolo11m.pt")

    # 2. Выполняем инференс на картинке
    # device="cuda:0" прямо указывает нейросети работать на видеокарте
    print("Запускаем инференс...")
    results = model("test_street.jpg", device = "cuda:0" if torch.cuda.is_available() else "cpu")

    # 3. Смотрим, что получилось
    # Библиотека Ultralytics умеет сама рисовать результаты для быстрой проверки!
    # Сохраним отрендеренную картинку на диск
    for result in results:
        # Метод plot() возвращает NumPy массив (картинку) с уже нарисованными рамками
        annotated_frame = result.plot()

        cv2.imwrite("test_street_result.jpg", annotated_frame)
        print("Готово! Результат сохранен в test_street_result.jpg")