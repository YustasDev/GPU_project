# core/streamer.py
import cv2
import logging
import queue
import threading
import time
from queue import Queue


logger = logging.getLogger(__name__)  # логгер с именем модуля (core.streamer)


class VideoStreamer(threading.Thread):  # поток, читающий кадры и кладущий их в очередь
    def __init__(self, source, frame_queue: Queue, reconnect_delay: float = 1.0):
        super().__init__(name="VideoStreamer", daemon=True)  # имя потока + daemon=True (умрёт с программой)
        self.source = source  # источник: индекс камеры, путь к файлу или RTSP URL
        self.frame_queue = frame_queue  # очередь, куда складываем свежие кадры
        self.reconnect_delay = reconnect_delay  # пауза перед попыткой переоткрыть источник, сек
        # Источник — локальный файл? (строка-путь, а не сетевой URL). Для файла потом
        # выдерживаем реальный FPS, иначе cap.read() «проглотит» всё видео за доли секунды.
        self._is_file = isinstance(source, str) and not source.startswith(
            ("rtsp://", "http://", "https://")
        )
        self._stop_event = threading.Event()  # потокобезопасный флаг остановки

    @property
    def stopped(self) -> bool:  # совместимость со старым публичным атрибутом stopped
        return self._stop_event.is_set()  # True, если остановку уже запросили

    @property
    def is_file(self) -> bool:  # источник — локальный видеофайл, а не камера/сетевой поток
        return self._is_file

    def stop(self) -> None:  # внешний API: попросить поток завершиться
        self._stop_event.set()  # выставляем событие — цикл выйдет на ближайшей проверке

    def _open_capture(self):  # вспомогательный метод: открыть VideoCapture с нужными настройками
        cap = cv2.VideoCapture(self.source)  # создаём объект захвата
        if cap.isOpened():  # если источник открылся успешно
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # минимальный внутренний буфер OpenCV (важно для RTSP/USB)
        return cap  # возвращаем cap; вызывающий сам проверит isOpened()

    def _push_frame(self, frame) -> None:  # неблокирующая укладка кадра в очередь
        try:
            self.frame_queue.put_nowait(frame)  # быстрая, не блокирующая попытка положить кадр
        except queue.Full:  # очередь полна — потребитель отстаёт
            try:
                self.frame_queue.get_nowait()  # выкидываем самый старый кадр (нам важен real-time)
            except queue.Empty:  # потребитель уже сам забрал — нормальная ситуация
                pass
            try:
                self.frame_queue.put_nowait(frame)  # повторная попытка положить свежий кадр
            except queue.Full:  # крайне маловероятно, но не блокируемся
                logger.debug("Очередь по-прежнему полна, кадр отброшен")  # отладочное сообщение

    def _frame_interval(self, cap) -> float:  # пауза между кадрами в зависимости от типа источника
        # Камера/RTSP сами задают темп выдачи кадров — лишь чуть притормаживаем,
        # чтобы отдать GIL потребителю и не жечь CPU на быстрых источниках.
        if not self._is_file:
            return 0.005  # 5 мс
        # Для видеофайла cap.read() отдаёт кадры максимально быстро, поэтому держим
        # паузу под реальный FPS файла, чтобы воспроизведение шло в нормальном темпе.
        fps = cap.get(cv2.CAP_PROP_FPS)  # частота кадров из метаданных файла
        if not fps or fps <= 0:  # некоторые контейнеры FPS не сообщают → тогда используем дефолтное значение 30 кадров/с
            fps = 30.0
        return 1.0 / fps  # время на один кадр

    def run(self) -> None:  # основной метод потока, вызывается из start()
        logger.info("Подключение к источнику: %s", self.source)  # стартовое сообщение
        cap = self._open_capture()  # первая попытка открыть источник
        frame_interval = self._frame_interval(cap)  # темп выдачи: реальный FPS файла или 5 мс для камеры
        try:
            while not self._stop_event.is_set():  # крутимся, пока не попросили остановиться
                if not cap.isOpened():  # источник закрыт — пробуем переоткрыть
                    logger.error(  # сообщаем об ошибке открытия
                        "Не удалось открыть источник %s, повтор через %.1f с",
                        self.source,
                        self.reconnect_delay,
                    )
                    if self._stop_event.wait(self.reconnect_delay):  # прерываемое ожидание
                        break  # за время ожидания пришёл stop() — выходим
                    cap = self._open_capture()  # повторная попытка открыть источник
                    continue  # к следующей итерации цикла

                ret, frame = cap.read()  # пробуем прочитать очередной кадр
                if not ret:  # кадр не получен
                    if self._is_file:  # для ФАЙЛА это конец видео — НЕ переоткрываем
                        logger.info("Видеофайл закончился (EOF), чтение остановлено.")
                        break  # выходим: конвейер доработает уже захваченные кадры
                    # камера/сетевой поток — это обрыв связи, пробуем переподключиться (как раньше)
                    logger.warning("Видеопоток прерван, попытка переподключиться")  # предупреждение
                    cap.release()  # освобождаем старый capture перед переоткрытием
                    if self._stop_event.wait(self.reconnect_delay):  # ждём, но прерываемо stop()
                        break  # пришёл stop() во время ожидания — выходим
                    cap = self._open_capture()  # пытаемся переоткрыть источник
                    continue  # к следующей итерации цикла

                self._push_frame(frame)  # кладём свежий кадр в очередь
                time.sleep(frame_interval)  # выдерживаем темп: реальный FPS файла либо 5 мс для камеры
        except Exception:  # ловим всё, чтобы гарантированно отпустить cap в finally
            logger.exception("Непредвиденная ошибка в VideoStreamer")  # лог со стектрейсом
        finally:
            if cap is not None:  # перестраховка: cap всегда определён, но проверим
                cap.release()  # освобождаем источник
            logger.info("Поток остановлен.")  # сообщаем о завершении работы потока
