# main.py
import logging  # для настройки root-логгера и единого формата сообщений
import time  # для пауз главного цикла и стартового grace-периода
from pathlib import Path  # абсолютные пути относительно расположения этого файла
from queue import Queue  # потокобезопасные очереди-«нити связи» между воркерами

from core.detector import AIDetector  # поток-детектор YOLO
from core.logger import DBLogger  # поток-писатель в PostgreSQL и на диск
from core.streamer import VideoStreamer  # поток-чтение кадров из источника


# === Конфигурация ===

PROJECT_DIR = Path(__file__).resolve().parent  # корень проекта; стабилен независимо от CWD
MODEL_PATH = PROJECT_DIR / "runs" / "detect" / "ai_runs" / "cow_learning" / "weights" / "best.pt"  # обученные веса cow+person
SAVE_DIR = PROJECT_DIR / "data" / "saved_events"  # каталог для JPEG-скриншотов событий
VIDEO_SOURCE = 0  # 0 = веб-камера; для теста можно подставить путь к .mp4 или RTSP URL
TARGET_CLASSES = [0, 1]  # 0=cow, 1=person по data.yaml в cow_dataset (см. CLAUDE.md)
FRAME_QUEUE_MAX = 30  # ~1 секунда буфера кадров на 30 fps
EVENT_QUEUE_MAX = 100  # запас событий на случай задержек БД
STARTUP_GRACE = 3.0  # сколько ждём после start() до проверки is_alive() (DB-схема, загрузка YOLO)
STREAMER_JOIN_TIMEOUT = 2.0  # стример реагирует на stop() почти мгновенно через _stop_event.wait
DETECTOR_JOIN_TIMEOUT = 2.0  # один проход YOLO-инференса заметно меньше секунды
DB_LOGGER_JOIN_TIMEOUT = 10.0  # больше — на случай дренажа очереди событий и медленных коммитов
STATUS_INTERVAL = 5.0  # период статусных строк с размерами очередей


logger = logging.getLogger(__name__)  # модульный логгер main для статуса и shutdown-сообщений


def configure_logging() -> None:
    # ОБЯЗАТЕЛЬНО вызывать в самом начале main(): без basicConfig у root-логгера
    # уровень WARNING и нет handler — все logger.info из streamer/detector/logger
    # тихо пропадут, и отлаживать систему станет невозможно.
    logging.basicConfig(
        level=logging.INFO,  # видеть «Сохранено: cow ...», «Загрузка модели...» и т.п.
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",  # время + уровень + имя модуля + текст
    )


def graceful_shutdown(streamer: VideoStreamer, detector: AIDetector, db_logger: DBLogger) -> None:
    # Гасим в порядке поток-данных: сначала продюсер, потом потребители.
    # Если остановить логгер первым, детектор продолжит толкать в event_queue, и будет race с _drain().
    steps = (
        (streamer, STREAMER_JOIN_TIMEOUT),  # сначала перестаём поставлять кадры
        (detector, DETECTOR_JOIN_TIMEOUT),  # затем детектор дорабатывает остаток frame_queue
        (db_logger, DB_LOGGER_JOIN_TIMEOUT),  # последним — логгер, чтобы он успел дренировать event_queue
    )
    for thread, timeout in steps:  # обрабатываем каждый поток последовательно (а не параллельно)
        if not thread.is_alive():  # поток уже умер (например, на этапе старта) — пропускаем
            continue
        thread.stop()  # просим завершиться (взводит _stop_event)
        thread.join(timeout=timeout)  # ждём окончания, но не бесконечно
        if thread.is_alive():  # таймаут истёк, поток ещё жив
            # Все потоки daemon=True, так что ОС снимет их при выходе процесса.
            # Логируем, чтобы оператор увидел проблему.
            logger.warning("Поток %s не завершился за %.1f с", thread.name, timeout)


def main() -> int:
    configure_logging()  # ПЕРВЫМ ДЕЛОМ — иначе нижние logger.info ничего не выведут
    logger.info("=== Запуск системы Smart Observer ===")  # видимый стартовый маркер
    logger.info("Модель: %s", MODEL_PATH)  # явно фиксируем, какие веса используем
    logger.info("Источник видео: %s", VIDEO_SOURCE)  # источник
    logger.info("Целевые классы: %s (0=cow, 1=person)", TARGET_CLASSES)  # расшифровка id

    if not MODEL_PATH.is_file():  # быстрый предполётный чек: веса должны существовать
        logger.error("Файл весов не найден: %s", MODEL_PATH)  # понятное сообщение оператору
        return 1  # ненулевой код возврата для shell/systemd/docker

    # Очереди — единственный способ обмена между потоками; maxsize ограничивает RAM
    frame_queue: Queue = Queue(maxsize=FRAME_QUEUE_MAX)  # стример → детектор
    event_queue: Queue = Queue(maxsize=EVENT_QUEUE_MAX)  # детектор → логгер

    # Воркеры; конструкторы лёгкие — реальная работа в .run() после .start()
    streamer = VideoStreamer(source=VIDEO_SOURCE, frame_queue=frame_queue)
    detector = AIDetector(
        model_path=str(MODEL_PATH),  # AIDetector принимает строку (передаётся в YOLO())
        frame_queue=frame_queue,  # откуда брать кадры
        event_queue=event_queue,  # куда класть события
        target_classes=TARGET_CLASSES,  # фильтруем детекции по этим классам
    )
    db_logger = DBLogger(
        event_queue=event_queue,  # переименован с logger -> db_logger, чтобы не затирать модульный logger
        save_dir=str(SAVE_DIR),  # абсолютный путь под JPEG, не зависит от CWD
    )

    # Порядок старта: сначала консумеры (чтобы были готовы принимать), потом продюсер.
    # Если запустить streamer первым, кадры пойдут в очередь, пока detector ещё не загрузил модель.
    db_logger.start()  # открывает сессию БД, делает create_all
    detector.start()  # загружает YOLO на GPU (это самое долгое — ~3–5 секунд)
    streamer.start()  # открывает камеру и начинает читать кадры

    # Health-check: даём потокам подняться и проверяем, что никто не умер на старте.
    # Ловит «быстрые» поломки: БД недоступна (create_all падает), нет файла весов и т.п.
    time.sleep(STARTUP_GRACE)  # пассивная пауза; не идеально, но достаточно для учебной системы
    dead = [t for t in (db_logger, detector, streamer) if not t.is_alive()]  # кто не дожил
    if dead:  # хотя бы один поток упал
        for t in dead:
            logger.error("Поток %s не запустился — выход", t.name)  # причину покажет лог в его run()
        graceful_shutdown(streamer, detector, db_logger)  # гасим тех, кто всё-таки поднялся
        return 1  # нештатное завершение

    logger.info("Все потоки запущены, входим в главный цикл (Ctrl+C для остановки).")

    try:
        while True:  # держим программу живой и периодически печатаем статус
            time.sleep(STATUS_INTERVAL)  # пауза между отчётами
            # qsize() даёт приблизительное значение под нагрузкой — для статуса этого достаточно.
            # Растёт frames — детектор узкое место; растёт events — БД узкое место.
            logger.info(
                "Очереди: frames=%d/%d events=%d/%d",
                frame_queue.qsize(),
                FRAME_QUEUE_MAX,
                event_queue.qsize(),
                EVENT_QUEUE_MAX,
            )
    except KeyboardInterrupt:  # Ctrl+C → SIGINT → KeyboardInterrupt в главном потоке
        logger.info("Получен Ctrl+C, начинаем штатное завершение...")  # видимое подтверждение

    graceful_shutdown(streamer, detector, db_logger)  # стандартный путь остановки и при Ctrl+C
    logger.info("=== Система Smart Observer безопасно выключена ===")
    return 0  # нормальный код возврата


if __name__ == "__main__":
    raise SystemExit(main())  # код возврата main() уходит в shell как exit code процесса
