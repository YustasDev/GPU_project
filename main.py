# main.py
import argparse  # для CLI-флага --source
import logging  # для настройки root-логгера и единого формата сообщений
import time  # для пауз главного цикла и стартового grace-периода
from pathlib import Path  # абсолютные пути относительно расположения этого файла
from queue import Queue  # потокобезопасные очереди-«нити связи» между воркерами

from core.analyzer import LLMAnalyzer  # фоновый поток-анализатор: очередь detection_id → LLM (троттлинг)
from core.detector import AIDetector  # поток-детектор YOLO
from core.logger import DBLogger  # поток-писатель в PostgreSQL и на диск
from core.streamer import VideoStreamer  # поток-чтение кадров из источника


# === Конфигурация ===

PROJECT_DIR = Path(__file__).resolve().parent  # корень проекта; стабилен независимо от CWD
MODEL_PATH = PROJECT_DIR / "best.pt"  # 2-классовый чекпоинт (person+car) с Roboflow Universe
SAVE_DIR = PROJECT_DIR / "data" / "saved_events"  # каталог для JPEG-скриншотов событий
LOG_DIR = PROJECT_DIR / "logs"  # каталог для файловых логов (info.log / warning.log / error.log)
# Порядок классов в скачанной модели может быть любым. После старта смотрите
# первую строку лога "Классы модели: ..." и подправьте список, если, например,
# в вашей модели 0=car, 1=person.
TARGET_CLASSES = [0, 1]
FRAME_QUEUE_MAX = 30  # ~1 секунда буфера кадров на 30 fps
EVENT_QUEUE_MAX = 100  # запас событий на случай задержек БД
ANALYSIS_QUEUE_MAX = 50  # запас detection_id для LLM-анализатора (он медленный из-за троттлинга)
STARTUP_GRACE = 3.0  # сколько ждём после start() до проверки is_alive() (DB-схема, загрузка YOLO)
STREAMER_JOIN_TIMEOUT = 2.0  # стример реагирует на stop() почти мгновенно через _stop_event.wait
DETECTOR_JOIN_TIMEOUT = 2.0  # один проход YOLO-инференса заметно меньше секунды
DB_LOGGER_JOIN_TIMEOUT = 10.0  # больше — на случай дренажа очереди событий и медленных коммитов
ANALYZER_JOIN_TIMEOUT = 5.0  # LLM-запрос может идти до ~30 c; дольше не ждём — поток daemon
STATUS_INTERVAL = 5.0  # период статусных строк с размерами очередей


logger = logging.getLogger(__name__)  # модульный логгер main для статуса и shutdown-сообщений


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smart Observer — детекция person и car в видеопотоке.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help=(
            "Источник видео: индекс веб-камеры (0, 1, ...), путь к видеофайлу "
            "или RTSP/HTTP URL. По умолчанию '0' (первая веб-камера)."
        ),
    )
    return parser.parse_args()


def resolve_source(value: str) -> int | str | None:
    # Цифровое значение — индекс веб-камеры (cv2.VideoCapture принимает int)
    if value.isdigit():
        return int(value)
    # Сетевые источники передаём как строку, файловой проверки тут не делаем
    if value.startswith(("rtsp://", "http://", "https://")):
        return value
    # Иначе считаем, что это путь к локальному файлу — валидируем
    path = Path(value)
    if path.is_file():
        return str(path)
    logger.error("Файл источника не найден: %s", value)
    return None


def configure_logging() -> None:
    # ОБЯЗАТЕЛЬНО вызывать в самом начале main(): без basicConfig у root-логгера
    # уровень WARNING и нет handler — все logger.info из streamer/detector/logger
    # тихо пропадут, и отлаживать систему станет невозможно.
    LOG_DIR.mkdir(parents=True, exist_ok=True)  # каталог под файлы логов; idempotent

    fmt = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"  # один формат для всех handler'ов

    def _level_file(level: int, name: str, *, or_higher: bool = False) -> logging.FileHandler:
        # Файл-handler: по умолчанию пропускает ТОЛЬКО свой уровень (строгое равенство),
        # чтобы info.log/warning.log оставались моноуровневыми. Для error.log используется
        # or_higher=True — он должен также собирать CRITICAL (более высокий уровень).
        h = logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")
        h.setLevel(level)  # отсекает всё ниже своего уровня ещё до фильтра
        if or_higher:
            h.addFilter(lambda r, lv=level: r.levelno >= lv)  # ERROR + CRITICAL
        else:
            h.addFilter(lambda r, lv=level: r.levelno == lv)  # ровно этот уровень
        return h

    logging.basicConfig(
        level=logging.INFO,  # видеть «Сохранено: cow ...», «Загрузка модели...» и т.п.
        format=fmt,
        handlers=[
            logging.StreamHandler(),                               # stderr — как было раньше
            _level_file(logging.INFO, "info"),                     # logs/info.log    — только INFO
            _level_file(logging.WARNING, "warning"),               # logs/warning.log — только WARNING
            _level_file(logging.ERROR, "error", or_higher=True),   # logs/error.log   — ERROR + CRITICAL
        ],
    )


def graceful_shutdown(
    streamer: VideoStreamer,
    detector: AIDetector,
    db_logger: DBLogger,
    analyzer: LLMAnalyzer,
) -> None:
    # Гасим в порядке поток-данных: сначала продюсер, потом потребители.
    # Если остановить логгер первым, детектор продолжит толкать в event_queue, и будет race с _drain().
    steps = (
        (streamer, STREAMER_JOIN_TIMEOUT),  # сначала перестаём поставлять кадры
        (detector, DETECTOR_JOIN_TIMEOUT),  # затем детектор дорабатывает остаток frame_queue
        (db_logger, DB_LOGGER_JOIN_TIMEOUT),  # затем логгер дренирует event_queue
        (analyzer, ANALYZER_JOIN_TIMEOUT),  # последним — LLM-анализатор (финальный потребитель detection_id)
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
    args = parse_args()  # читаем CLI до запуска потоков
    source = resolve_source(args.source)  # int | str | None
    if source is None:
        return 1  # ошибка уже залогирована в resolve_source

    logger.info("=== Запуск системы Smart Observer ===")  # видимый стартовый маркер
    logger.info("Модель: %s", MODEL_PATH)  # явно фиксируем, какие веса используем
    logger.info("Источник видео: %s", source)  # тип int (вебка) или str (файл/URL)
    logger.info("Целевые классы: %s", TARGET_CLASSES)  # имена будут видны из лога модели

    if not MODEL_PATH.is_file():  # быстрый предполётный чек: веса должны существовать
        logger.error("Файл весов не найден: %s", MODEL_PATH)  # понятное сообщение оператору
        return 1  # ненулевой код возврата для shell/systemd/docker

    # Очереди — единственный способ обмена между потоками; maxsize ограничивает RAM
    frame_queue: Queue = Queue(maxsize=FRAME_QUEUE_MAX)  # стример → детектор
    event_queue: Queue = Queue(maxsize=EVENT_QUEUE_MAX)  # детектор → логгер
    analysis_queue: Queue = Queue(maxsize=ANALYSIS_QUEUE_MAX)  # логгер → LLM-анализатор (detection_id)

    # Воркеры; конструкторы лёгкие — реальная работа в .run() после .start()
    streamer = VideoStreamer(source=source, frame_queue=frame_queue)
    detector = AIDetector(
        model_path=str(MODEL_PATH),  # AIDetector принимает строку (передаётся в YOLO())
        frame_queue=frame_queue,  # откуда брать кадры
        event_queue=event_queue,  # куда класть события
        target_classes=TARGET_CLASSES,  # фильтруем детекции по этим классам
    )
    db_logger = DBLogger(
        event_queue=event_queue,  # переименован с logger -> db_logger, чтобы не затирать модульный logger
        save_dir=str(SAVE_DIR),  # абсолютный путь под JPEG, не зависит от CWD
        analysis_queue=analysis_queue,  # сюда логгер кладёт id сохранённых строк для LLM-анализа
    )
    analyzer = LLMAnalyzer(analysis_queue=analysis_queue)  # один поток на все LLM-запросы, с троттлингом

    # Порядок старта: сначала консумеры (чтобы были готовы принимать), потом продюсер.
    # Если запустить streamer первым, кадры пойдут в очередь, пока detector ещё не загрузил модель.
    db_logger.start()  # открывает сессию БД, делает create_all
    analyzer.start()  # поток-потребитель detection_id; ждёт, пока появятся сохранённые события
    detector.start()  # загружает YOLO на GPU (это самое долгое — ~3–5 секунд)
    streamer.start()  # открывает камеру и начинает читать кадры

    # Health-check: даём потокам подняться и проверяем, что никто не умер на старте.
    # Ловит «быстрые» поломки: БД недоступна (create_all падает), нет файла весов и т.п.
    time.sleep(STARTUP_GRACE)  # пассивная пауза; не идеально, но достаточно для учебной системы
    dead = [t for t in (db_logger, analyzer, detector, streamer) if not t.is_alive()]  # кто не дожил
    if dead:  # хотя бы один поток упал
        for t in dead:
            logger.error("Поток %s не запустился — выход", t.name)  # причину покажет лог в его run()
        graceful_shutdown(streamer, detector, db_logger, analyzer)  # гасим тех, кто всё-таки поднялся
        return 1  # нештатное завершение

    logger.info("Все потоки запущены, входим в главный цикл (Ctrl+C для остановки).")

    try:
        while True:  # держим программу живой и периодически печатаем статус
            time.sleep(STATUS_INTERVAL)  # пауза между отчётами
            # qsize() даёт приблизительное значение под нагрузкой — для статуса этого достаточно.
            # Растёт frames — детектор узкое место; растёт events — БД узкое место.
            logger.info(
                "Очереди: frames=%d/%d events=%d/%d analysis=%d/%d",
                frame_queue.qsize(),
                FRAME_QUEUE_MAX,
                event_queue.qsize(),
                EVENT_QUEUE_MAX,
                analysis_queue.qsize(),
                ANALYSIS_QUEUE_MAX,
            )
    except KeyboardInterrupt:  # Ctrl+C → SIGINT → KeyboardInterrupt в главном потоке
        logger.info("Получен Ctrl+C, начинаем штатное завершение...")  # видимое подтверждение

    graceful_shutdown(streamer, detector, db_logger, analyzer)  # стандартный путь остановки и при Ctrl+C
    logger.info("=== Система Smart Observer безопасно выключена ===")
    return 0  # нормальный код возврата


if __name__ == "__main__":
    raise SystemExit(main())  # код возврата main() уходит в shell как exit code процесса
