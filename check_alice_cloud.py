import base64, os, httpx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- НАСТРОЙКИ: замените на свои значения ---
YANDEX_API_KEY = os.environ.get("YANDEX_API_KEY")       # API-ключ сервисного аккаунта
YANDEX_FOLDER_ID = "b1gna7vvqq60g9jnf65l"               # Идентификатор каталога (folder_id)
IMAGE_PATH = "break open2.jpg"                          # Путь к файлу картинки на локальном ПК
MODEL_NAME = "qwen3.6-35b-a3b/latest"
PROMPT_TEXT = (
    "Ты — строгий охранник системы видеонаблюдения. "
    "Посмотри на это изображение. Кратко опиши человека или машину, которых ты видишь. "
    "Укажи, есть ли что-то подозрительное на изображении. "
    "Отвечай на русском языке, максимум 2 предложения."
)
# ------------------------------------------


def image_to_base64(image_path: str) -> str:
    """Читает файл и возвращает строку в формате Base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_image_with_alice():
    # Инициализация клиента (OpenAI-совместимый API Yandex Cloud)
    client = OpenAI(
        api_key=YANDEX_API_KEY,
        base_url="https://ai.api.cloud.yandex.net/v1",
        project=YANDEX_FOLDER_ID,  # Передаётся как project (это folder_id)
        # trust_env=False — игнорировать ALL_PROXY/*_PROXY из окружения.
        # Тот же приём применён в core/analyzer.py.
        http_client=httpx.Client(trust_env=False),
    )

    # Кодирование картинки
    image_base64 = image_to_base64(IMAGE_PATH)

    # Формирование запроса к мультимодальной модели Alice AI LLM
    response = client.chat.completions.create(
        model=f"gpt://{YANDEX_FOLDER_ID}/{MODEL_NAME}",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT_TEXT
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "high"  # auto / low / high — влияет на детализацию и стоимость
                        }
                    }
                ]
            }
        ],
        # qwen3.6-35b-a3b — reasoning-модель: часть токенов уходит в reasoning_content
        # (цепочку рассуждений). Бюджет должен покрыть И рассуждения, И финальный ответ,
        # иначе ответ обрывается по лимиту (finish_reason="length") и content == None.
        max_tokens=5000,
        temperature=0.7
    )

    # Извлечение и вывод ответа
    answer = response.choices[0].message.content  # choices — это список; берём первый вариант
    print("Ответ модели:")
    print(answer)

if __name__ == "__main__":
    analyze_image_with_alice()