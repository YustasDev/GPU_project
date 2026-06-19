import os, base64, httpx
from openai import OpenAI


if __name__ == "__main__":

    client = OpenAI(base_url="https://models.github.ai/inference",
                    api_key=os.environ["GITHUB_TOKEN"],
                    http_client=httpx.Client(trust_env=False))

    with open("break open2.jpg", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    resp = client.chat.completions.create(
      model="openai/gpt-4.1",
      messages=[{"role": "user", "content": [
          {"type": "text", "text": "Ты - охранник. Опиши внешность человека/людей на фото и оцени, подозрительно ли его поведение. Ответь одним коротким предложением."},
          {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
      ]}],
         )
    print(resp.choices[0].message.content)

