from openai import OpenAI

TEMPERATURE = 0
MAX_TOKENS = 512

API_KEY = ""
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

def get_api_responses(message):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    response = client.chat.completions.create(
        model = MODEL,
        messages = [
            {"role": "user", "content": message},
        ],
        temperature = TEMPERATURE,
        max_tokens = MAX_TOKENS,
        stream = False
    )
    return response.choices[0].message.content
