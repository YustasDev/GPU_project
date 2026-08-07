import os, base64
from openai import OpenAI
import textwrap
from dotenv import load_dotenv
import httpx

# Load the .env file
load_dotenv()

image = "./break open2.jpg"
router_key = os.getenv("ROUTER_API_KEY")
BASE_URL = "https://routerai.ru/api/v1"
MODEL = "google/gemma-4-26b-a4b-it"
GUARD_PROMPT = ("Ты - строгий охранник системы видеонаблюдения. Посмотри на изображение. "
                "Кратко опиши человека или машину, которых ты видишь. Укажи, есть ли что-то подозрительное на изображении. "
                "Отвечай на русском языке, максимум 2 предложения.")

if __name__ == "__main__":

    # 1. Инициализация клиента
    client = OpenAI(
        base_url=BASE_URL,
        api_key=router_key,
        http_client=httpx.Client(trust_env=False)
    )

    # 2. Кодирование изображения в base64
    with open(image, "rb") as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')

    # 3. Вызов модели
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=2.0,
        top_p=0.95,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": GUARD_PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    }
                ]
            }
        ]
    )

    # 4. Вывод результата
    answer = resp.choices[0].message.content
    print(textwrap.fill(answer, width=100))