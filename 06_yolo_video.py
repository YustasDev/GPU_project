import cv2
import math
import time
from ultralytics import YOLO
from videostream import FileVideoStream
from drawer import draw_detection

# НАСТРОЙКИ
INPUT_SOURCE = "traffic.mp4"  # 0 для веб-камеры, или "traffic.mp4"
OUTPUT_VIDEO = "yolo_output.mp4"
TARGET_CLASSES = [0, 2]  # Будем искать ТОЛЬКО: 0 (человек), 2 (машина). Остальное игнорируем!


def main():
    print("[INFO] Загрузка YOLO11m на GPU...")
    model = YOLO("yolo11m.pt")
    class_names = model.names

    print("[INFO] Запуск видеопотока...")
    fvs = FileVideoStream(INPUT_SOURCE)

    # Читаем свойства видео ДО запуска фонового потока, чтобы избежать race condition
    width = int(fvs.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(fvs.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = fvs.stream.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None or math.isnan(fps): fps = 30.0

    fvs.start()
    time.sleep(1.0)  # Прогрев камеры

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    print("[INFO] Начинаем обработку. Ctrl+C для остановки.")
    try:
        while True:
            if not fvs.more() and fvs.stopped:
                break

            if fvs.more():
                frame = fvs.read()
                if frame is None:
                    continue
                frame_count += 1

                # --- ОБРАБОТКА ИНФЕРЕНСА ---
                # Отдаем кадр нейросети. conf=0.5 означает "игнорировать все, где уверенность < 50%"
                results = model(frame, device="cuda:0", verbose=False, conf=0.5)
                result = results[0]

                # Извлекаем данные профилирования (скорость работы) из объекта results
                speed = result.speed
                preprocess_ms = speed['preprocess']
                inference_ms = speed['inference']
                postprocess_ms = speed['postprocess']
                total_ms = preprocess_ms + inference_ms + postprocess_ms

                # Парсинг результатов и отрисовка
                for box in result.boxes:
                    class_id = int(box.cls[0].item())

                    # Фильтр: рисуем рамку только если это человек или машина
                    if class_id in TARGET_CLASSES:
                        conf = box.conf[0].item()
                        coords = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                        class_name = class_names[class_id]

                        # Вызываем нашу рисовалку
                        draw_detection(frame, (x1, y1, x2, y2), class_name, conf)

                # Отрисовка телеметрии на кадре
                cv2.putText(frame, f"Pre-process: {preprocess_ms:.1f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 0), 2)
                cv2.putText(frame, f"Inference (GPU): {inference_ms:.1f} ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)
                cv2.putText(frame, f"Post-process: {postprocess_ms:.1f} ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2)

                # Расчет реального FPS для этого кадра
                current_fps = 1000.0 / total_ms if total_ms > 0 else 0
                cv2.putText(frame, f"Net FPS: {current_fps:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                            2)

                out.write(frame)

                # Ограничение для веб-камеры (10 секунд)
                if isinstance(INPUT_SOURCE, int) and frame_count >= 300:
                    break
            else:
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[ОСТАНОВКА] Прервано пользователем.")
    finally:
        fvs.stop()
        out.release()
        print(f"[INFO] Сохранено в {OUTPUT_VIDEO}. Обработано кадров: {frame_count}")


if __name__ == "__main__":
    main()