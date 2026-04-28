from ultralytics import YOLO
import cv2

if __name__ == "__main__":

    model = YOLO("/home/yustasdev/AI_project/runs/detect/ai_runs/cow_learning/weights/best.pt")

    image_path = "76338724.jpg"
    output_path = "result_detected.jpg"

    # verbose=False отключает вывод в консоль подробной информации
    results = model(image_path, device="cuda:0", verbose=False)

    # У нас одна картинка, берем первый результат
    result = results[0]

    # model.names - это словарь. Ключ: ID класса (0), Значение: Имя ('person')
    # посмотрим, какие классы объектов существуют в модели
    classes_dict = model.names
    print(classes_dict)

    # Загружаем оригинальное изображение, чтобы рисовать поверх него
    image = cv2.imread(image_path)

    # Цвета рамок для каждого класса (BGR, потому что OpenCV)
    colors = {
       0: (0, 0, 255),    # person - красный
       1: (0, 255, 0),    # cow - зелёный
    }

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

        # 4. Рисуем рамку вокруг найденного объекта
        color = colors.get(class_id, (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 5. Готовим подпись: имя класса + уверенность в процентах
        label = f"{class_name} {conf * 100:.1f}%"

        # Считаем размер текста, чтобы нарисовать под него заполненный фон
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # Фон под текст рисуем над рамкой (чтобы текст был читаемым на любой картинке)
        cv2.rectangle(image, (x1, y1 - text_h - baseline - 4), (x1 + text_w, y1), color, -1)

        # Сам текст пишем чёрным поверх цветного фона
        cv2.putText(image, label, (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Сохраняем картинку с нарисованными рамками
    cv2.imwrite(output_path, image)
    print(f"\n[INFO] Картинка с рамками сохранена: {output_path}")