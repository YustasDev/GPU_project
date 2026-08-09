# core/analyzer.py
import base64
import logging
import os
import queue
import threading
from functools import lru_cache

import httpx
from dotenv import load_dotenv
from openai import OpenAI

from db.database import SessionLocal  # фабрика сессий БД
from db.models import Detection  # ORM-модель таблицы detections

logger = logging.getLogger(__name__)  # логгер модуля (core.analyzer) — пишет в общие logs/*.log

# === Константы (вынесены, чтобы не хардкодить по месту) ===
BASE_URL = "https://routerai.ru/api/v1"  # эндпоинт RouterAI (OpenAI-совместимый)
# Активная vision-модель. ВАЖНО: она обязана принимать изображения (быть мультимодальной) —
# текстовой модели картинка вернётся ошибкой. Сменить модель = поменять ровно эту одну строку.
MODEL = "google/gemma-4-26b-a4b-it"  # мультимодальная Gemma 4 — та же, что в тест-скрипте gemma4_vision.py
REQUEST_TIMEOUT = 30.0  # сек: ограничиваем сетевой запрос, чтобы фоновый поток не висел вечно
TEMPERATURE = 1.5  # «температура»: чем выше, тем разнообразнее формулировки (как в gemma4_vision.py)
TOP_P = 0.95  # ядерная выборка: слова берутся из верхушки, накопившей 95% вероятности
MAX_TOKENS = 500  # потолок длины ответа — бережёт квоту и не даёт переполнить колонку
MAX_DESCRIPTION_LEN = 500  # длина колонки Detection.description (VARCHAR(500))
ANALYSIS_THROTTLE_SEC = 3.0  # минимальный интервал между запросами к LLM (~20 в минуту) — бережём лимиты провайдера

PROMPT_TEXT = (
    "Ты — строгий охранник системы видеонаблюдения. "
    "Посмотри на это изображение. Кратко опиши человека или машину, которых ты видишь. "
    "Укажи, есть ли что-то подозрительное на изображении. "
    "Отвечай на русском языке, максимум 2 предложения."
)


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    """Ленивое создание клиента: при ПЕРВОМ вызове, а не на импорте модуля.
    Так отсутствие ROUTER_API_KEY не уронит весь пайплайн (модуль импортируется в main.py)."""
    load_dotenv()  # подхватываем .env, если есть
    token = os.environ.get("ROUTER_API_KEY")  # читаем ключ RouterAI из окружения
    if not token:  # нет ключа — сообщаем понятной ошибкой (ловится в analyze_event)
        raise RuntimeError("ROUTER_API_KEY не задан — определите его в .env или окружении")
    return OpenAI(
        base_url=BASE_URL,
        api_key=token,
        http_client=httpx.Client(trust_env=False),  # игнорировать ALL_PROXY/*_PROXY из окружения
        timeout=REQUEST_TIMEOUT,  # таймаут на запрос — поток не зависнет на сетевой проблеме
    )


def encode_image(image_path: str) -> str:
    """Превращает картинку в строку Base64 для data-URL."""
    with open(image_path, "rb") as image_file:  # бинарное чтение файла
        return base64.b64encode(image_file.read()).decode("utf-8")  # bytes -> base64 -> str


def analyze_event(detection_id: int) -> None:
    """Запрашивает у LLM описание события #detection_id и пишет его в Detection.description.
    Безопасна для запуска из фонового потока: все сбои логируются, исключения наружу не летят."""
    try:
        client = _get_client()  # ленивая инициализация (бросит RuntimeError, если нет токена)
    except RuntimeError as e:  # без токена просто пропускаем анализ, не роняя поток/пайплайн
        logger.error("LLM-анализ события #%s пропущен: %s", detection_id, e)
        return

    with SessionLocal() as db:  # отдельная сессия для потока-анализатора
        # 1. Находим событие по ID
        event = db.query(Detection).filter(Detection.id == detection_id).first()
        if not event or not event.image_path:  # нет строки или картинки — анализировать нечего
            return
        if event.description:  # уже проанализировано — не тратим квоту повторно (идемпотентность)
            return

        # 2. Кодируем картинку (файл мог быть удалён/перемещён — обрабатываем отдельно)
        try:
            base64_image = encode_image(event.image_path)
        except OSError as e:  # FileNotFoundError, PermissionError и т.п.
            logger.warning("Не удалось прочитать %s: %s", event.image_path, e)
            return

        logger.info("[LLM] анализ события #%s...", detection_id)  # старт анализа в лог

        # 3. Запрос к мультимодальной модели + сохранение результата
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,  # те же параметры сэмплинга, что и в тест-скрипте gemma4_vision.py
                top_p=TOP_P,  # ядерная выборка — отсекаем «хвост» маловероятных слов
                max_tokens=MAX_TOKENS,  # ограничиваем длину ответа
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT_TEXT},
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }
                ],
            )
            text = (response.choices[0].message.content or "").strip()  # ответ модели (без пробелов по краям)
            event.description = text[:MAX_DESCRIPTION_LEN]  # обрезаем под VARCHAR(500), иначе Postgres бросит ошибку
            db.commit()  # сохраняем описание в БД
            logger.info("[LLM] событие #%s: %s", detection_id, event.description)
        except Exception:  # сетевые сбои, 429 (rate-limit), ошибки БД и пр. — поток не роняем
            logger.exception("[LLM] ошибка анализа события #%s", detection_id)
            db.rollback()  # откат «грязной» транзакции (например, при сбое commit)


class LLMAnalyzer(threading.Thread):  # ОДИН фоновый поток-потребитель detection_id (вместо «поток на событие»)
    """Берёт detection_id из очереди и обрабатывает их по одному, с троттлингом между запросами,
    чтобы не превышать лимиты провайдера. По шаблону AIDetector/DBLogger (stop()/_stop_event)."""

    def __init__(
        self,
        analysis_queue: queue.Queue,  # очередь detection_id, которую наполняет DBLogger
        throttle_sec: float = ANALYSIS_THROTTLE_SEC,  # мин. пауза между запросами к LLM
        get_timeout: float = 0.5,  # таймаут ожидания id — даёт прерываемость stop()
    ):
        super().__init__(name="LLMAnalyzer", daemon=True)  # имя потока + daemon=True (умрёт с программой)
        self.analysis_queue = analysis_queue  # источник detection_id
        self.throttle_sec = throttle_sec  # троттлинг под rate-limit провайдера
        self.get_timeout = get_timeout  # период пробуждения цикла при пустой очереди
        self._busy = False  # занят ли анализатор обработкой события прямо сейчас (для авто-стопа в main)
        self._stop_event = threading.Event()  # потокобезопасный флаг остановки

    @property
    def stopped(self) -> bool:  # единый публичный атрибут, как у других потоков
        return self._stop_event.is_set()

    @property
    def busy(self) -> bool:  # True, пока идёт обработка одного события (main ждёт этого перед авто-стопом)
        return self._busy

    def stop(self) -> None:  # внешний API: попросить поток завершиться
        self._stop_event.set()  # цикл выйдет на ближайшей проверке; прервёт и троттлинг-ожидание

    def run(self) -> None:  # основной метод потока
        logger.info("LLM-анализатор запущен (троттлинг %.1f c между запросами).", self.throttle_sec)
        while not self._stop_event.is_set():  # крутимся, пока не попросили остановиться
            try:
                detection_id = self.analysis_queue.get(timeout=self.get_timeout)  # ждём id с таймаутом
            except queue.Empty:  # за таймаут ничего не пришло — проверим _stop_event и снова ждём
                continue
            self._busy = True  # помечаем занятость: main не остановит нас на полузапросе
            try:
                analyze_event(detection_id)  # сам ловит свои ошибки (сеть, 429, БД) и не бросает наружу
            except Exception:  # подстраховка от чего-либо непредвиденного — поток не должен умирать
                logger.exception("LLM-анализатор: непредвиденная ошибка для #%s", detection_id)
            finally:
                self._busy = False  # обработка события завершена
            # Троттлинг: пауза между запросами, прерываемая по stop() (wait вернётся сразу при set()).
            self._stop_event.wait(self.throttle_sec)
        logger.info("LLM-анализатор остановлен.")  # финальное сообщение о завершении потока
