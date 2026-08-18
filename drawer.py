import cv2

# Цвета в формате BGR (Blue, Green, Red)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)


def draw_detection(image, box, label, confidence, color=COLOR_GREEN):
    """
    Рисует красивый бокс с подложкой под текст.
    box: (x1, y1, x2, y2) - координаты (int)
    label: str - название класса (например, "person")
    confidence: float - уверенность (0.0 - 1.0)
    """
    x1, y1, x2, y2 = box

    # 1. Рисуем сам прямоугольник
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)

    # 2. Готовим текст
    text = f"{label} {confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1

    # Узнаем размер текста, чтобы сделать под него фон
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # 3. Рисуем подложку (залитый прямоугольник) над боксом
    # Если бокс у самого верха, рисуем подложку внутри бокса
    if y1 - text_h - 10 > 0:
        bg_rect_pt1 = (x1, y1 - text_h - 10)
        bg_rect_pt2 = (x1 + text_w, y1)
        text_pt = (x1, y1 - 5)
    else:
        bg_rect_pt1 = (x1, y1)
        bg_rect_pt2 = (x1 + text_w, y1 + text_h + 10)
        text_pt = (x1, y1 + text_h + 5)

    cv2.rectangle(image, bg_rect_pt1, bg_rect_pt2, color, thickness=-1)  # -1 заливка

    # 4. Пишем белый текст поверх цветной подложки
    cv2.putText(image, text, text_pt, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)