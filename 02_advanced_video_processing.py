import cv2
import time
import signal
import threading
from videostream import FileVideoStream  # Импортируем наш класс многопоточного чтения

# НАСТРОЙКА ИСТОЧНИКА
# Вариант А: Путь к файлу (например, "traffic.mp4")
# Вариант Б: Индекс устройства (как правило: 0 - встроенная веб-камера, 1 - USB-камера)
INPUT_SOURCE = 0  # Попробуйте изменить на "traffic.mp4", чтобы увидеть разницу
OUTPUT_VIDEO = "output_fast.mp4"

# НАСТРОЙКА НАДПИСЕЙ ПОВЕРХ КАДРА
FONT = cv2.FONT_HERSHEY_SIMPLEX  # шрифт надписей
FONT_SCALE = 0.6  # размер надписей: 1.0 — крупно, 0.6 — компактно
FONT_THICKNESS = 2  # толщина линий шрифта

stop_event = threading.Event()   # определяем флаг прерывания процесса

def signal_handler(signum, frame):
    """Обработчик сигнала для прерывания"""
    print("\n[ВНИМАНИЕ] Процесс прерван пользователем (Ctrl+C).")
    stop_event.set()  # устанавливаем флаг прерывания


def process_video_fast():
    print(f"[INFO] Запуск захвата видео из источника: {INPUT_SOURCE}")

    # Устанавливаем обработчик сигнала. Важно: после этой строки Python больше НЕ
    # возбуждает KeyboardInterrupt по Ctrl+C — сигнал целиком уходит в наш обработчик,
    # поэтому ловить прерывание еще и через "except KeyboardInterrupt" не нужно
    signal.signal(signal.SIGINT, signal_handler)

    # Живой источник (веб-камера или сетевой поток) сам никогда не заканчивается.
    # Отличаем его один раз и дальше пользуемся готовым ответом
    is_live_source = isinstance(INPUT_SOURCE, int) or (
        isinstance(INPUT_SOURCE, str) and "://" in INPUT_SOURCE)

    # Запускаем многопоточный ридер
    fvs = FileVideoStream(INPUT_SOURCE).start()

    # Даем камере секунду на "прогрев" и наполнение буфера. Видеофайл в прогреве
    # не нуждается — он готов отдавать кадры сразу
    if is_live_source:
        time.sleep(1.0)

    # Метаданные берем у самого ридера: он прочитал их один раз при открытии
    # источника и уже применил поправку для камер, которые врут про FPS
    width, height, fps = fvs.width, fvs.height, fvs.fps

    print(f"[INFO] Разрешение потока: {width}x{height} @ {fps} FPS.")

    # Настраиваем "Писатель" (VideoWriter)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    # Проверяем, что писатель действительно открылся. Без этой проверки программа
    # честно отработает до конца и отчитается об успехе, а на диске останется
    # пустой файл — самая обидная разновидность ошибки
    if not out.isOpened():
        fvs.stop()
        raise IOError(f"Не удалось открыть файл для записи: {OUTPUT_VIDEO!r}")

    # Ширину надписи LIVE REC вычисляем один раз: так индикатор аккуратно встанет
    # в правый верхний угол при любом разрешении кадра
    (live_width, _), _ = cv2.getTextSize("LIVE REC", FONT, FONT_SCALE, FONT_THICKNESS)

    frame_count = 0
    start_time = time.time()

    # Переменные для вычисления реального FPS (скорости обработки)
    fps_start_time = time.time()
    current_fps = 0.0

    print("[INFO] Начинаем обработку. Нажмите Ctrl+C в терминале для экстренной остановки.")

    try:
        while not stop_event.is_set():   # проверяем, установлен ли флаг прерывания
            # Если очередь пуста, а поток завершил чтение (конец файла) — выходим
            if not fvs.more() and fvs.stopped:
                print("[INFO] Видеофайл закончился.")
                break

            # Если в буфере есть готовые кадры, забираем их
            if fvs.more():
                frame = fvs.read()
                if frame is None:  # очередь опустела между more() и read()
                    continue
                frame_count += 1

                # Камера иногда отдает кадр не того размера, который сама же заявила
                # в своих свойствах. VideoWriter в этом случае молча не пишет ничего,
                # поэтому приводим кадр к размеру, с которым открыт писатель
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))

                # --- НАЧАЛО БЛОКА ОБРАБОТКИ ---

                # Вычисляем текущий FPS обработки (обновляем каждые 30 кадров)
                if frame_count % 30 == 0:
                    end_time = time.time()
                    current_fps = 30 / (end_time - fps_start_time)
                    fps_start_time = time.time()
                    # Выводим прогресс в консоль отдельной строкой: так по окончании
                    # работы виден весь график скорости, а не только последний замер
                    print(f"[ПРОГРЕСС] Обработано: {frame_count:>4} кадров | Скорость: {current_fps:.1f} FPS")
                elif frame_count < 30:
                    # Первые 30 кадров еще не набрались. Показываем среднюю скорость
                    # с начала работы, иначе на коротком ролике в кадр попадет "0.0"
                    current_fps = frame_count / (time.time() - start_time)

                # Рисуем зеленый счетчик FPS в левом верхнем углу (как в играх)
                cv2.putText(frame, f"Proc FPS: {current_fps:.1f}", (20, 35),
                            FONT, FONT_SCALE, (0, 255, 0), FONT_THICKNESS, cv2.LINE_AA)

                # Рисуем красный индикатор LIVE REC, если источник живой (камера или поток)
                if is_live_source:
                    cv2.putText(frame, "LIVE REC", (width - live_width - 20, 35),
                                FONT, FONT_SCALE, (0, 0, 255), FONT_THICKNESS, cv2.LINE_AA)
                    # Красная точка слева от надписи
                    cv2.circle(frame, (width - live_width - 35, 30), 6, (0, 0, 255), -1)

                # --- КОНЕЦ БЛОКА ОБРАБОТКИ ---

                # Сохраняем обработанный кадр в файл
                out.write(frame)

                # ЗАЩИТА ОТ БЕСКОНЕЧНОГО ЦИКЛА ДЛЯ ЖИВОГО ИСТОЧНИКА
                # Ни камера, ни сетевой поток не заканчиваются сами,
                # поэтому запишем только первые 300 кадров (10 секунд)
                if is_live_source and frame_count >= 300:
                    print("[INFO] Записано 300 кадров с живого источника. Остановка.")
                    break
            else:
                # Если очередь пуста (процессор работает быстрее, чем камера/диск отдает кадры),
                # усыпляем процесс на 1 миллисекунду, чтобы не грузить CPU на 100%.
                time.sleep(0.001)

    finally:
        # 4. Освобождаем ресурсы (выполнится в любом случае, даже при ошибке)
        print("[INFO] Очистка памяти и закрытие файлов...")
        fvs.stop()
        out.release()

        total_time = time.time() - start_time
        print(f"[INFO] Успешно завершено! Результат сохранен в {OUTPUT_VIDEO}.")
        print(f"[INFO] Общее время работы: {total_time:.2f} сек.")


if __name__ == "__main__":
    process_video_fast()
