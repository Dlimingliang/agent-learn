import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

class LlmClient:
        def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = 30, stream: bool = False):
            self.model = model
            self.apiKey = apiKey
            self.baseUrl = baseUrl
            self.timeout = timeout
            self.stream = stream
            self.headers: dict[str, str] = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.apiKey}'
            }
        def chat(self, messages: list[dict[str, str]], temperature: int = 0):
            data = {
                'model': self.model,
                'messages': messages,
                'temperature': temperature,
            }
            url = f'{self.baseUrl}/chat/completions'
            response = requests.post(url, json=data, headers=self.headers, timeout=self.timeout)
            print(f"ResponseType: {type(response)}")
            print(f"Response: {response}")
            print(f"ResponseJson: {response.json()}")

class OpenAiClient:
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = 30):
        self.model = model
        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)
    def chat(self, messages: list[dict[str, str]], temperature: int = 0, stream: bool = False):
        """
         调用大语言模型进行思考，并返回其响应。
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=stream)
            print(f"ResponseType: {type(response)}")
            print(f"Response: {response}")
            print(f"ResponseJson: {response.model_dump_json()}")
        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")


if __name__ == '__main__':
    model = os.getenv("LLM_MODEL_ID")
    apiKey = os.getenv("LLM_API_KEY")
    baseUrl = os.getenv("LLM_BASE_URL")
    messages = [
                {"role": "user", "content": "你好，请简单介绍一下你自己。"}
            ]
    openai_client = OpenAiClient(model=model, apiKey=apiKey, baseUrl=baseUrl)
    openai_client.chat(messages=messages)