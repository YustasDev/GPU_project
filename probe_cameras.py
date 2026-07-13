# probe_cameras.py — перебор индексов камер: какой реально отдаёт кадр.
# Полезно, когда камера создаёт несколько узлов /dev/video* и неясно, какой из них
# настоящий поток захвата (RGB). Запуск (окружение ai_project активировано):
#   python probe_cameras.py
import os

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"  # приглушаем предупреждения V4L2 (иначе много мусора в консоли)

import cv2  # импорт ПОСЛЕ выставления OPENCV_LOG_LEVEL, иначе уровень не подхватится

MAX_INDEX = 10  # сколько индексов проверить: /dev/video0 .. /dev/video9


def main() -> None:
    print("Проверяю индексы камер...\n")
    working = []  # сюда соберём индексы, реально отдавшие кадр
    for i in range(MAX_INDEX):
        # CAP_V4L2 — явно просим бэкенд V4L2, чтобы индекс i соответствовал /dev/video{i}
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if not cap.isOpened():  # узел не открылся — пропускаем
            cap.release()
            continue
        ok, frame = cap.read()  # НАСТОЯЩАЯ проверка: пришёл ли кадр (isOpened() недостаточно)
        if ok and frame is not None:
            h, w = frame.shape[:2]
            print(f"  [OK] индекс {i} (/dev/video{i}): кадр {w}x{h}  <- рабочая камера")
            working.append(i)
        else:
            print(f"  [--] индекс {i} (/dev/video{i}): открылся, но кадр не пришёл")
        cap.release()  # ОБЯЗАТЕЛЬНО освобождаем устройство перед следующей итерацией

    print()
    if working:
        print(f"Рабочие индексы: {working} -> используйте --source {working[0]}")
    else:
        print("Кадр не отдал ни один индекс (камера подключена? права на /dev/video*?).")


if __name__ == "__main__":
    main()
