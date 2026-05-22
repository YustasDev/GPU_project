# core/detector.py
import logging
import queue
import threading
from queue import Queue
import cv2
from ultralytics import YOLO

logger = logging.getLogger(__name__)  # логгер с именем модуля (core.detector)


class AIDetector(threading.Thread):  # поток: потребитель кадров и продюсер событий
    def __init__(
        self,
        model_path: str,  # путь к весам или имя модели (например 'yolo11s.pt')
        frame_queue: Queue,  # очередь, откуда забираем кадры от VideoStreamer
        event_queue: Queue,  # очередь, куда отдаём события для писателя в БД
        target_classes,  # iterable id классов, которые считаем «целевыми»
        device: str = "cuda:0",  # устройство инференса; по конвенции проекта — первая GPU
        confidence: float = 0.5,  # порог уверенности YOLO для включения детекции
        get_timeout: float = 0.1,  # таймаут ожидания кадра — даёт прерываемость stop()
    ):
        super().__init__(name="AIDetector", daemon=True)  # имя потока + daemon=True (умрёт с программой)
        self.model_path = model_path  # сохраняем для лога и загрузки модели в run()
        self.frame_queue = frame_queue  # источник кадров
        self.event_queue = event_queue  # сток событий
        self.target_classes = set(target_classes)  # set: O(1) проверка `in` и защита от дубликатов
        self.device = device  # устройство для model(...)
        self.confidence = confidence  # порог conf для YOLO
        self.get_timeout = get_timeout  # период проверки _stop_event при пустой очереди
        self._stop_event = threading.Event()  # потокобезопасный флаг остановки (как в VideoStreamer)

    @property
    def stopped(self) -> bool:  # совместимость со старым публичным атрибутом stopped
        return self._stop_event.is_set()  # True, если уже попросили остановиться

    def stop(self) -> None:  # внешний API: попросить поток завершиться
        self._stop_event.set()  # выставляем событие — цикл выйдет на ближайшей проверке

    def _emit_event(self, event_data: dict) -> None:  # неблокирующая отправка события в event_queue
        try:
            self.event_queue.put_nowait(event_data)  # быстрая неблокирующая попытка положить элемент
        except queue.Full:  # очередь переполнена — писатель в БД отстаёт
            logger.warning(  # фиксируем потерю события вместо тихого молчания
                "event_queue полна, событие %s отброшено",
                event_data.get("object_class"),
            )

    def run(self) -> None:  # основной метод потока, вызывается из start()
        logger.info("Загрузка модели %s на устройство %s...", self.model_path, self.device)  # стартовое сообщение
        try:
            model = YOLO(self.model_path)  # загружаем веса (Ultralytics сам подкачает при необходимости)
            class_names = model.names  # сопоставление class_id -> имя класса для подписи и события
            logger.info("Классы модели: %s", model.names)  # пользователь сверит TARGET_CLASSES в main.py
            logger.info("Модель готова к инференсу.")  # сообщаем, что готовы потреблять кадры
        except Exception:  # ошибка загрузки модели (нет файла, некорректные веса, нет CUDA и т.п.)
            logger.exception("Не удалось загрузить модель %s", self.model_path)  # лог со стектрейсом
            return  # выходим — main узнает по is_alive() и сможет среагировать

        try:
            while not self._stop_event.is_set():  # крутимся, пока не попросили остановиться
                try:
                    frame = self.frame_queue.get(timeout=self.get_timeout)  # блокирующее ожидание с таймаутом
                except queue.Empty:  # кадр не пришел — проверим _stop_event и снова ждём
                    continue

                try:
                    results = model(  # синхронный инференс на выбранном устройстве
                        frame,
                        device=self.device,  # параметр из __init__
                        verbose=False,  # выключаем подробный вывод Ultralytics в stdout
                        conf=self.confidence,  # порог уверенности
                    )
                except Exception:  # ошибка инференса (OOM, некорректный кадр, отвалилась CUDA и др.)
                    logger.exception("Ошибка инференса; кадр пропущен")  # пропускаем кадр, поток продолжает работу
                    continue  # к следующему кадру

                result = results[0]  # обрабатываем один кадр — берём первый и единственный результат

                # Собираем все целевые детекции в список ДО отрисовки.
                # Без этого ранние события содержали бы кадр без поздних рамок (кумулятивный баг).
                detections = []  # элементы: (class_id, conf, x1, y1, x2, y2)
                for box in result.boxes:  # перебираем все боксы, найденные на кадре
                    class_id = int(box.cls[0].item())  # id класса как Python int
                    if class_id not in self.target_classes:  # неинтересный класс — пропускаем
                        continue
                    conf = float(box.conf[0].item())  # уверенность как Python float
                    coords = box.xyxy[0].cpu().numpy()  # координаты [x1, y1, x2, y2] в numpy на CPU
                    x1, y1, x2, y2 = (  # приводим к int для функций cv2
                        int(coords[0]),
                        int(coords[1]),
                        int(coords[2]),
                        int(coords[3]),
                    )
                    detections.append((class_id, conf, x1, y1, x2, y2))  # сохраняем для второго прохода

                if not detections:  # на кадре нет интересующих объектов
                    continue  # событий не формируем — к следующему кадру

                # Одна аннотированная копия на весь кадр — её получат ВСЕ события этого кадра.
                # Так все события одного кадра содержат одинаковую (полную) разметку.
                annotated = frame.copy()  # копия один раз на кадр, а не на каждый бокс
                for class_id, conf, x1, y1, x2, y2 in detections:  # рисуем все целевые боксы
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)  # красная рамка, толщина 2
                    label = f"{class_names[class_id]} {conf:.2f}"  # подпись вида «person 0.87»
                    cv2.putText(  # подпись над рамкой
                        annotated,
                        label,
                        (x1, y1 - 10),  # 10 пикселей выше верхней границы рамки
                        cv2.FONT_HERSHEY_SIMPLEX,  # стандартный шрифт OpenCV
                        0.5,  # масштаб шрифта
                        (0, 0, 255),  # красный — под цвет рамки
                        2,  # толщина линий текста
                    )

                # По одному событию на детекцию. Ключи словаря совпадают с колонками
                # db.models.Detection — это прямой контракт для писателя в БД.
                for class_id, conf, x1, y1, x2, y2 in detections:  # формируем события
                    event_data = {
                        "object_class": class_names[class_id],  # DB column: object_class
                        "confidence": conf,  # DB column: confidence
                        "bounding_box": f"[{x1}, {y1}, {x2}, {y2}]",  # DB column: bounding_box (String(100))
                        "frame": annotated,  # numpy-кадр; писатель сохранит на диск и проставит image_path
                    }
                    self._emit_event(event_data)  # неблокирующая отправка в event_queue
        except Exception:  # любая необработанная ошибка в основном цикле
            logger.exception("Непредвиденная ошибка в AIDetector")  # не теряем стектрейс
        finally:
            logger.info("Детектор остановлен.")  # финальное сообщение о завершении потока
