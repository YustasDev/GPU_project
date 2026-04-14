from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("yolo11m.pt")

    # verbose=False отключает вывод в консоль подробной информации
    results = model("test_street.jpg", device="cuda:0", verbose=False)

    # У нас одна картинка, берем первый результат
    result = results[0]

    # model.names - это словарь. Ключ: ID класса (0), Значение: Имя ('person')
    # посмотрим, какие классы объектов существуют в модели
    classes_dict = model.names
    print(classes_dict)

    print("\n--- РАСШИФРОВКА ДЕТЕКЦИЙ ---")
    # Все найденные рамки лежат в объекте result.boxes
    for box in result.boxes:
        # 1. Извлекаем координаты xyxy (x_min, y_min, x_max, y_max)
        # .cpu() переносит тензор из VRAM в RAM
        # .numpy() превращает тензор PyTorch в массив NumPy
        coords = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

        # 2. Извлекаем уверенность (Confidence)
        # .item() вытаскивает одно число из тензора
        conf = box.conf[0].item()

        # 3. Извлекаем класс объекта
        class_id = int(box.cls[0].item())
        class_name = classes_dict[class_id]

        print(f"Нашли: {class_name:10} | Уверенность: {conf * 100:.1f}% | Координаты:[{x1}, {y1}, {x2}, {y2}]")