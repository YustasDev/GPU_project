# check_ollama_cloud.py
# Зеркало LLM-вызова из core/analyzer.py, но через Ollama.
#
# Идея главы: КОД, который разговаривает с моделью, один и тот же — меняется лишь,
# КУДА он обращается. Ollama даёт OpenAI-совместимый API, поэтому клиент — тот же OpenAI SDK,
# что и в analyzer.py. Здесь показаны два пути к облачной модели Ollama Cloud:
#
#   • ЛОКАЛЬНЫЙ ДЕМОН (по умолчанию):  base_url = http://localhost:11434/v1
#       Демон Ollama проксирует :cloud-модель в облако. Авторизация — это разовый
#       `ollama signin` НА СТОРОНЕ ДЕМОНА; клиентский код креды НЕ передаёт (api_key —
#       заглушка "ollama", которую требует OpenAI SDK; демон сам подставит ваши облачные креды).
#       Этот же base_url обслуживает и полностью локальную модель (см. MODEL) — код не меняется.
#
#   • ПРЯМОЙ, БЕЗ ДЕМОНА:  base_url = https://ollama.com/v1
#       Идем в облако Ollama напрямую. Демон/`ollama signin` не нужны; ключ OLLAMA_API_KEY
#       читаем из .env — ровно также как analyzer.py читает OPENROUTER_API_KEY.
#
# ── Что нужно один раз сделать (bootstrap, в код не входит) ──────────────────
#   Для локального пути:  установить Ollama >= 0.22, запустить демон (ollama serve),
#                          и для :cloud-моделей — ollama signin.
#   Для прямого пути:      положить OLLAMA_API_KEY=... в .env (ключ с ollama.com → Settings → API keys).
#
# ── Какую модель используем (MODEL) ─────────────────────────────────────────────
#   • Облако:   "gemma4:31b-cloud"  → считает облако Ollama (это 31B; MoE 26b-a4b в облаке НЕ хостится).
#   • Локально: "gemma4:26b"        → та самая 26b-a4b MoE на ВАШЕЙ GPU, офлайн, без signin.


import base64
import os

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # подхватываем .env (для прямого пути нужен OLLAMA_API_KEY)

# === Выбор пути к облаку ===
USE_DIRECT_CLOUD = True   # False → локальный демон (localhost); True → прямой ollama.com (ключ из .env)

if USE_DIRECT_CLOUD:
    BASE_URL = "https://ollama.com/v1"           # прямой облачный эндпоинт (демон не нужен)
    API_KEY = os.environ.get("OLLAMA_API_KEY")   # ключ из .env — как OPENROUTER_API_KEY в analyzer.py
    if not API_KEY:
        raise RuntimeError("OLLAMA_API_KEY не задан — укажите его в .env "
                           "(сгенерируйте: ollama.com → Settings → API keys)")
else:
    BASE_URL = "http://localhost:11434/v1"       # локальный демон Ollama (OpenAI-совместимый)
    API_KEY = "ollama"                            # заглушка: демон авторизован через `ollama signin`,
                                                  # клиентский код облачные креды НЕ передаёт

MODEL = "gemma4:31b-cloud"                        # облако (=31B). Локально на своей GPU: "gemma4:26b"
IMAGE_PATH = "break open2.jpg"
MAX_TOKENS = 600
REQUEST_TIMEOUT = 120.0                           # облачный 31B отвечает небыстро — но не ждём вечно

PROMPT_TEXT = (
    "Ты — строгий охранник системы видеонаблюдения. "
    "Посмотри на это изображение. Кратко опиши человека или машину, которых ты видишь. "
    "Укажи, есть ли что-то подозрительное на изображении. "
    "Отвечай на русском языке, максимум 2 предложения."
)


def encode_image(image_path: str) -> str:
    """Читает файл и возвращает строку Base64 для data-URL."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


if __name__ == "__main__":
    base64_image = encode_image(IMAGE_PATH)

    # Клиент — тот же, что и в core/analyzer.py: OpenAI SDK + настраиваемый base_url + timeout.
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        http_client=httpx.Client(trust_env=False),  # игнорировать ALL_PROXY/*_PROXY (обход VPN socks://)
        timeout=REQUEST_TIMEOUT,                     # не подвисаем молча навсегда
    )

    # Блок запроса с картинкой — БЕЗ изменений относительно analyzer.py (base64 data-URL).
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
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

    # Аккуратный разбор ответа (как в check_openrouter_model.py)
    text = (response.choices[0].message.content or "").strip()
    print("Ответ модели:")
    print(text if text else "(пустой content — проверьте, поддерживает ли модель vision)")
