from openai import OpenAI

def no_think_test(model: str = None, apiKey: str = None, baseUrl: str = None,
                  user_input: str = None, temperature: int = 0, stream: bool = False):
    test_client = OpenAI(api_key=apiKey, base_url=baseUrl)
    messages = [
        {"role": "system", "content": "有用的助手"},
        {"role": "user", "content": user_input}
    ]
    response = test_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=stream
    )
    print(response.model_dump_json())

def think_test(model: str = None, apiKey: str = None, baseUrl: str = None,
                  user_input: str = None, temperature: int = 0, stream: bool = False):
    test_client = OpenAI(api_key=apiKey, base_url=baseUrl)
    messages = [
        {"role": "system", "content": "有用的助手"},
        {"role": "user", "content": user_input}
    ]
    response = test_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=stream,
        extra_body={"thinking": {"type": "enabled"}}
    )
    print(response.model_dump_json())

if __name__ == '__main__':
    model = ""
    url = ""
    apikey = ""
    user_input ="我想学习agent开发，请为我制定一个为期4周的学习计划，包括每周的重点内容和实践项目"
    no_think_test(model=model, apiKey=apikey, baseUrl=url, user_input=user_input)
    think_model = "deepseek-reasoner"
    think_test(model=think_model, apiKey=apikey, baseUrl=url, user_input=user_input)