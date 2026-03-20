import cv2
import time
import signal
import threading
from videostream import FileVideoStream  # Импортируем наш класс многопоточного чтения

# НАСТРОЙКА ИСТОЧНИКА
# Вариант А: Путь к файлу (например, "traffic.mp4")
# Вариант Б: Индекс устройства (0 - встроенная веб-камера, 1 - USB-камера)
INPUT_SOURCE = 0  # Попробуйте изменить на "traffic.mp4", чтобы увидеть разницу
OUTPUT_VIDEO = "output_fast.mp4"

stop_event = threading.Event()

def signal_handler(signum, frame):
    """Обработчик сигнала для прерывания"""
    print("\n[ВНИМАНИЕ] Процесс прерван пользователем (Ctrl+C).")
    stop_event.set()


def process_video_fast():
    print(f"[INFO] Запуск захвата видео из источника: {INPUT_SOURCE}")

    # 1. Устанавливаем обработчик сигнала
    signal.signal(signal.SIGINT, signal_handler)

    # 1. Запускаем многопоточный ридер
    fvs = FileVideoStream(INPUT_SOURCE).start()

    # Даем потоку 1 секунду на "прогрев" камеры и заполнение буфера
    time.sleep(1.0)

    # 2. Получаем метаданные видео напрямую из оригинального объекта stream
    width = int(fvs.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(fvs.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = fvs.stream.get(cv2.CAP_PROP_FPS)

    # ХАК ДЛЯ ВЕБ-КАМЕР: Часто камеры возвращают fps = 0 или NaN.
    # Если это произошло, принудительно ставим стандартные 30 кадров в секунду.
    if fps == 0 or fps is None or fps != fps:
        fps = 30.0

    print(f"[INFO] Разрешение потока: {width}x{height} @ {fps} FPS.")

    # 3. Настраиваем "Писатель" (VideoWriter)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    # Переменные для вычисления реального FPS (скорости обработки)
    fps_start_time = time.time()
    current_fps = 0.0

    print("[INFO] Начинаем обработку. Нажмите Ctrl+C в терминале для экстренной остановки.")

    try:
        while not stop_event.is_set():
            # Если очередь пуста, а поток завершил чтение (конец файла) — выходим
            if not fvs.more() and fvs.stopped:
                print("\n[INFO] Видеофайл закончился.")
                break

            # Если в буфере есть готовые кадры, забираем их
            if fvs.more():
                frame = fvs.read()
                frame_count += 1

                # --- НАЧАЛО БЛОКА ОБРАБОТКИ ---

                # Вычисляем текущий FPS обработки (обновляем каждые 30 кадров)
                if frame_count % 30 == 0:
                    end_time = time.time()
                    current_fps = 30 / (end_time - fps_start_time)
                    fps_start_time = time.time()
                    # Выводим прогресс в консоль
                    print(f"\r[ПРОГРЕСС] Обработано: {frame_count} кадров | Скорость: {current_fps:.1f} FPS", end="")

                # Рисуем зеленый счетчик FPS в левом верхнем углу (как в играх)
                cv2.putText(frame, f"Proc FPS: {current_fps:.1f}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Рисуем красный индикатор LIVE REC, если источник - веб-камера (число)
                if isinstance(INPUT_SOURCE, int):
                    cv2.putText(frame, "LIVE REC", (width - 170, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.circle(frame, (width - 190, 40), 10, (0, 0, 255), -1)  # Красная точка

                # --- КОНЕЦ БЛОКА ОБРАБОТКИ ---

                # Сохраняем обработанный кадр в файл
                out.write(frame)

                # ЗАЩИТА ОТ БЕСКОНЕЧНОГО ЦИКЛА ДЛЯ ВЕБ-КАМЕРЫ
                # Так как веб-камера не имеет конца, мы запишем только первые 300 кадров (10 секунд)
                if isinstance(INPUT_SOURCE, int) and frame_count >= 300:
                    print("\n[INFO] Записано 300 кадров с веб-камеры. Остановка.")
                    break
            else:
                # Если очередь пуста (процессор работает быстрее, чем камера/диск отдает кадры),
                # усыпляем процесс на 1 миллисекунду, чтобы не грузить CPU на 100%.
                time.sleep(0.001)

    except KeyboardInterrupt:
        # Обработка нажатия Ctrl+C пользователем
        print("\n[ВНИМАНИЕ] Процесс прерван пользователем.")

    finally:
        # 4. Освобождаем ресурсы (выполнится в любом случае, даже при ошибке)
        print("\n[INFO] Очистка памяти и закрытие файлов...")
        fvs.stop()
        out.release()

        total_time = time.time() - start_time
        print(f"[INFO] Успешно завершено! Результат сохранен в {OUTPUT_VIDEO}.")
        print(f"[INFO] Общее время работы: {total_time:.2f} сек.")


if __name__ == "__main__":
    process_video_fast()