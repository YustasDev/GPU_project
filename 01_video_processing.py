import cv2
import time

# Пути к файлам
INPUT_VIDEO = "traffic.mp4"  # Замените на свой файл с видео
OUTPUT_VIDEO = "output.mp4"


def process_video():
    # Открываем захват видео
    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Получаем метаданные видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {width}x{height} @ {fps} FPS. Total frames: {total_frames}")

    # Настраиваем "Писатель" (Writer) для сохранения результата
    # Codec 'mp4v' обычно работает везде. Для веба лучше 'avc1' (h.264), но требует OpenH264
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    while True:
        # Читаем кадр. ret - успешность (True/False), frame - картинка (NumPy array)
        ret, frame = cap.read()

        if not ret:
            print("Video finished or error.")
            break

        # --- ЗДЕСЬ БУДЕТ МАГИЯ AI (пока просто инвертируем цвета) ---
        # Обработка: превратим в негатив (демонстрация попиксельной операции)
        # frame = cv2.bitwise_not(frame)
        # (Закомментировано, чтобы сохранить оригинальные цвета для след. примера)

        # Просто нарисуем счетчик кадров
        cv2.putText(frame, f"Frame: {frame_count}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Записываем обработанный кадр в файл
        out.write(frame)

        frame_count += 1

        # Прогресс бар в консоли (чтобы не скучать)
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")

    # Освобождаем ресурсы (ОЧЕНЬ ВАЖНО)
    cap.release()
    out.release()

    duration = time.time() - start_time
    print(f"Done! Saved to {OUTPUT_VIDEO}. Time: {duration:.2f}s")


if __name__ == "__main__":
    process_video()