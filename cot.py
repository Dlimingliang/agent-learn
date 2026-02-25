import json
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv
import os

# 加载 .env 文件中的环境变量
load_dotenv()

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
        return response

class ChainOfThought:
    def __init__(self, client: OpenAiClient):
        self.client = client
    def solve_step_by_step(self, user_input: str) -> dict[str, Any]:
        # 第一步分解问题
        decomposition_prompt = f"""
            请将问题分解为3-5个逻辑步骤:
            问题: {user_input}
            
            请返回JSON格式:
            {{
                "steps":[
                    {{"step": 1, "description": "步骤描述", "question": "需要回答的具体问题"}},
                ...
                ]
            }}
        """
        messages = [{"role": "user", "content": decomposition_prompt}]
        response = self.client.chat(messages)
        try:
            steps_data = json.loads(response.choices[0].message.content)
            steps = steps_data.get("steps", [])
        except json.JSONDecodeError:
            # 备用方案：简单分解
            steps = [
                {"step": 1, "description": "理解问题", "question": f"如何理解这个问题：{user_input}？"},
                {"step": 2, "description": "分析解决方案", "question": "有哪些可能的解决方案？"},
                {"step": 3, "description": "得出结论", "question": "最佳答案是什么？"}
            ]
        print(f"🔗 问题已分解为 {len(steps)} 个步骤")

        # 逐步解决问题
        step_results = []
        context = f"原始问题：{user_input}\n\n"
        for step in steps:
            step_num = step["step"]
            description = step["description"]
            question = step["question"]
            print(f"  📍 步骤 {step_num}: {description}")
            step_prompt = f"""
                        {context}
                        当前步骤：{description}
                        具体问题：{question}

                        请基于前面的分析，详细回答这个步骤的问题。
                        """
            step_message = [{"role": "user", "content": step_prompt}]
            step_response = self.client.chat(step_message)
            step_result = step_response.choices[0].message.content
            step_results.append({
                "step": step_num,
                "description": description,
                "question": question,
                "answer": step_result
            })
            # 更新上下文
            context += f"步骤{step_num} ({description}): {step_result}\n\n"

        # 最终综合
        final_prompt = f"""
            基于以下逐步分析，请给出对原始问题的最终综合答案：

            {context}

            请提供一个清晰、完整的最终答案。
            """
        final_response = self.client.chat(messages=[{"role": "user", "content": final_prompt}])
        final_answer = final_response.choices[0].message.content
        return {
            "problem": user_input,
            "steps": step_results,
            "final_answer": final_answer,
            "total_steps": len(steps)
        }


if __name__ == '__main__':
    model = os.getenv("LLM_MODEL_ID")
    apiKey = os.getenv("LLM_API_KEY")
    baseUrl = os.getenv("LLM_BASE_URL")
    messages = [
        {"role": "user", "content": "一个班级有30个学生，其中60%是女生。如果新来了5个男生，现在男生和女生的比例是多少？"}
    ]
    llm_client = OpenAiClient(model=model, apiKey=apiKey, baseUrl=baseUrl)
    llm_client.chat(messages=messages)

    math_problem = "一个班级有30个学生，其中60%是女生。如果新来了5个男生，现在男生和女生的比例是多少？"
    cot = ChainOfThought(client=llm_client)
    cot_result = cot.solve_step_by_step(math_problem)
    print(f"📝 问题: {cot_result['problem']}")
    print(f"🔢 分解步骤数: {cot_result['total_steps']}")

    print("\n📋 详细步骤:")
    for step in cot_result['steps']:
        print(f"  {step['step']}. {step['description']}")
        print(f"     问题: {step['question']}")
        print(f"     答案: {step['answer'][:100]}...")
        print()

    print(f"🎯 最终答案: {cot_result['final_answer'][:200]}...")

    print("\n" + "=" * 70 + "\n")
