import requests
import json
import os
import base64
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
PROMPT_TEXT = (
    "Ты — строгий охранник системы видеонаблюдения. "
    "Посмотри на это изображение. Кратко опиши человека или машину, которых ты видишь. "
    "Укажи, есть ли что-то подозрительное на изображении. "
    "Отвечай на русском языке, максимум 2 предложения."
)

# Функция для конвертации локального файла в base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

if __name__ == "__main__":

    # Путь к вашей картинке на компьютере
    image_path = "break open2.jpg"
    base64_image = encode_image(image_path)

    response = requests.post(
      url="https://openrouter.ai/api/v1/chat/completions",
      headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
      },
      data=json.dumps({
        # ВАЖНО: модель должна поддерживать image-input (быть vision/мультимодальной).
        # Текстовые модели (напр. openai/gpt-oss-20b:free) на картинку вернут
        # 404 "No endpoints found that support image input".
        # Бесплатный каталог меняется — сверяйте актуальный :free-id на openrouter.ai/models.
        "model": "google/gemma-4-26b-a4b-it:free",
        "max_tokens": 200,
        "messages": [
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
                    # Передаем картинку в формате Data URL.
                    # Если у вас PNG, замените image/jpeg на image/png
                    "url": f"data:image/jpeg;base64,{base64_image}"
                  }
                }
              ]
            }
          ]
      })
    )

    # Разбираем ответ аккуратно: сначала проверяем статус и наличие 'choices',
    # иначе печатаем настоящее сообщение сервера (а не невнятный KeyError).
    data = response.json()
    if response.status_code == 200 and "choices" in data:
        print(data["choices"][0]["message"]["content"])
    else:
        # Ошибка API: показать код и текст, чтобы было понятно, что пошло не так
        err = data.get("error", data)
        print(f"Ошибка OpenRouter (HTTP {response.status_code}): {err}")