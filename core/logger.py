# core/logger.py
import logging
import os
import queue
import threading
from datetime import datetime, timezone
from queue import Queue

import cv2
from sqlalchemy.exc import SQLAlchemyError  # общий корень исключений SQLAlchemy

from db.database import Base, SessionLocal, engine  # схема, фабрика сессий и движок БД
from db.models import Detection  # ORM-модель таблицы detections

logger = logging.getLogger(__name__)  # логгер с именем модуля (core.logger)


class DBLogger(threading.Thread):  # поток-консумер событий: пишет JPEG на диск и строку в БД
    def __init__(
        self,
        event_queue: Queue,  # очередь событий от AI Detector
        save_dir: str,  # каталог для JPEG-снимков
        analysis_queue: Queue | None = None,  # очередь detection_id для LLM-анализатора (None = без LLM)
        get_timeout: float = 0.5,  # таймаут ожидания события — даёт прерываемость stop()
    ):
        super().__init__(name="DBLogger", daemon=True)  # имя потока + daemon=True (умрёт с программой)
        self.event_queue = event_queue  # источник событий
        self.save_dir = save_dir  # сохраняем для использования в _save_image
        self.analysis_queue = analysis_queue  # сюда кладём id сохранённых строк для LLM-анализа
        self.get_timeout = get_timeout  # период пробуждения цикла при пустой очереди
        self._stop_event = threading.Event()  # потокобезопасный флаг остановки (как в VideoStreamer/AIDetector)
        self._last_frame = None  # ссылка на последний обработанный numpy-кадр
        self._last_filepath = None  # путь к JPEG этого кадра (или None при сбое записи)
        os.makedirs(self.save_dir, exist_ok=True)  # создаём каталог заранее; если уже есть — ОК

    @property
    def stopped(self) -> bool:  # совместимость со старым публичным атрибутом stopped
        return self._stop_event.is_set()  # True, если кто-то уже вызвал stop()

    def stop(self) -> None:  # внешний API: попросить поток завершиться
        self._stop_event.set()  # выставляем событие — цикл выйдет на ближайшей проверке

    def _save_image(self, frame, object_class: str) -> str | None:  # запись кадра на диск
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")  # UTC + микросекунды
        filename = f"event_{object_class}_{timestamp_str}.jpg"  # вид: event_person_20260508_120033_123456.jpg
        filepath = os.path.join(self.save_dir, filename)  # полный путь к файлу
        if not cv2.imwrite(filepath, frame):  # imwrite возвращает False (не исключение) при сбое
            logger.warning("cv2.imwrite вернул False для %s — кадр не записан", filepath)  # фиксируем потерю
            return None  # пути нет — в БД пойдёт image_path=NULL (колонка nullable)
        return filepath  # успех — возвращаем путь для записи в БД

    def _handle_event(self, db, event: dict) -> None:  # обработка одного события: файл + строка БД
        frame = event["frame"]  # numpy-кадр с уже отрисованными рамками от детектора
        if frame is not self._last_frame:  # сверка по идентичности — новый кадр пишем один раз
            self._last_filepath = self._save_image(frame, event["object_class"])  # записываем JPEG
            self._last_frame = frame  # держим ссылку на кадр — иначе Python может переиспользовать id()
        try:
            record = Detection(  # ORM-объект под одну строку таблицы detections
                object_class=event["object_class"],  # имя класса (DB column: object_class)
                confidence=event["confidence"],  # уверенность (DB column: confidence)
                bounding_box=event["bounding_box"],  # строка "[x1, y1, x2, y2]" (DB column: bounding_box)
                image_path=self._last_filepath,  # путь к JPEG или None при сбое imwrite
            )
            db.add(record)  # ставим объект в сессию (INSERT произойдёт при flush/commit)
            db.commit()  # фиксируем транзакцию — строка появляется в БД, record.id заполняется
        except SQLAlchemyError:  # ловим любые ошибки SQLAlchemy: коннект, констрейнты, и т.п.
            logger.exception("Ошибка записи в БД; rollback и пропускаем событие")  # лог со стектрейсом
            db.rollback()  # откатываем транзакцию — иначе сессия залипает в «грязном» состоянии
            return  # это событие потеряно; следующее обработаем в чистой сессии
        logger.info(  # успех — фиксируем понятную строку в лог
            "Сохранено: %s (%.2f) -> id=%s, path=%s",
            event["object_class"],
            event["confidence"],
            record.id,  # primary key, проставленный БД при INSERT
            self._last_filepath,
        )

        # Отдаём id сохранённой строки ОДНОМУ фоновому LLM-анализатору (а не плодим потоки).
        # put_nowait — не блокируем писателя; если очередь полна (анализатор отстаёт из-за
        # троттлинга/лимитов) — пропускаем анализ этого события с предупреждением.
        if self.analysis_queue is not None:
            try:
                self.analysis_queue.put_nowait(record.id)
            except queue.Full:
                logger.warning("analysis_queue полна — LLM-анализ события id=%s пропущен", record.id)

    def _drain(self, db) -> None:  # дописать оставшиеся события после stop()
        drained = 0  # счётчик обработанных при дренаже событий
        while True:  # тянем из очереди всё, пока не пусто
            try:
                event = self.event_queue.get_nowait()  # без блокировки — иначе зависнем после stop()
            except queue.Empty:  # очередь пуста — выходим
                break
            self._handle_event(db, event)  # тот же путь обработки, что и в основном цикле
            drained += 1  # учитываем сохранённое
        if drained:  # печатаем итог только если что-то было дописано
            logger.info("При остановке дописано событий: %d", drained)

    def run(self) -> None:  # основной метод потока, вызывается из start()
        logger.info("Служба записи в БД запускается...")  # стартовое сообщение
        try:
            Base.metadata.create_all(bind=engine)  # создаём таблицы (idempotent — пропустит существующие)
        except SQLAlchemyError:  # БД может быть недоступна (docker не поднят), неправильные логин/пароль и т.п.
            logger.exception("Не удалось инициализировать схему БД, поток завершается")  # лог со стеком
            return  # выходим — main узнает по is_alive() и сможет среагировать

        logger.info("Схема БД готова, ждём события.")  # подтверждаем готовность

        try:
            with SessionLocal() as db:  # одна сессия на жизнь потока, автозакрытие при выходе
                while not self._stop_event.is_set():  # крутимся, пока не попросили остановиться
                    try:
                        event = self.event_queue.get(timeout=self.get_timeout)  # блокирующее ожидание
                    except queue.Empty:  # за таймаут события не пришло — проверим _stop_event и снова ждём
                        continue  # без sleep: get(timeout=...) уже корректно отдаёт GIL
                    self._handle_event(db, event)  # обработка одного события (файл + БД + запуск LLM)
                self._drain(db)  # после stop() дописываем хвост — события не теряются
        except Exception:  # любая необработанная ошибка вне SQLAlchemyError
            logger.exception("Непредвиденная ошибка в DBLogger")  # сохраняем стектрейс
        finally:
            logger.info("Служба записи в БД остановлена.")  # финальное сообщение о завершении потока
