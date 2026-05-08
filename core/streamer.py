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
        self._stop_event = threading.Event()  # потокобезопасный флаг остановки

    @property
    def stopped(self) -> bool:  # совместимость со старым публичным атрибутом stopped
        return self._stop_event.is_set()  # True, если остановку уже запросили

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

    def run(self) -> None:  # основной метод потока, вызывается из start()
        logger.info("Подключение к источнику: %s", self.source)  # стартовое сообщение
        cap = self._open_capture()  # первая попытка открыть источник
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
                if not ret:  # кадр не получен (конец файла или обрыв связи)
                    logger.warning("Видеопоток прерван, попытка переподключиться")  # предупреждение
                    cap.release()  # освобождаем старый capture перед переоткрытием
                    if self._stop_event.wait(self.reconnect_delay):  # ждём, но прерываемо stop()
                        break  # пришёл stop() во время ожидания — выходим
                    cap = self._open_capture()  # пытаемся переоткрыть источник
                    continue  # к следующей итерации цикла

                self._push_frame(frame)  # кладём свежий кадр в очередь
                time.sleep(0.005)  # 5 мс: отдаём GIL потребителю и не жжём CPU на быстрых источниках
        except Exception:  # ловим всё, чтобы гарантированно отпустить cap в finally
            logger.exception("Непредвиденная ошибка в VideoStreamer")  # лог со стектрейсом
        finally:
            if cap is not None:  # перестраховка: cap всегда определён, но проверим
                cap.release()  # освобождаем источник
            logger.info("Поток остановлен.")  # сообщаем о завершении работы потока
