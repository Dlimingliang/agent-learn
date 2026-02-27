import json
import os
import time
from dataclasses import field, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import requests
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

class LlmClient:
        def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = 600, stream: bool = False):
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
            return response

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

class AgentState(Enum):
    """Agent状态枚举"""
    IDLE = "idle" # 空闲
    PERCEIVING = "perceiving" # 感知
    PLANNING = "planning" # 规划
    ACTING = "acting" # 执行
    REFLECTING = "reflecting" # 反思
    ERROR = "error" # 错误

@dataclass
class AgentMemory:
    """Agent 记忆模块"""
    short_memory: list[dict[str, Any]] = field(default_factory=list)
    long_memory: dict[str, Any] = field(default_factory=dict)
    working_memory: dict[str, Any] = field(default_factory=dict)

    def add_short_memory(self, memory: dict[str, Any]):
        memory["timestamp"] = datetime.now().isoformat()
        self.short_memory.append(memory)
        if len(self.short_memory) > 10:
            # 限制短期记忆大小
            self.short_memory.pop(0)

    def update_working_memory(self, key: str, value: Any):
        self.working_memory[key] = value

    def get_context(self) -> str:
        context_parts = []
        if self.working_memory:
            context_parts.append(f"当前状态:{json.dumps(self.working_memory, ensure_ascii=False, indent=2)}")
        if self.short_memory:
            recent_memories = self.short_memory[-3:] # 最近的三条记忆
            memory_context = "\n".join([f"- {mem.get('content',mem)}"for mem in recent_memories])
            context_parts.append(memory_context)
        return "\n\n".join(context_parts)

@dataclass
class Task:
    id: str
    description: str
    priority: int = 1
    status: str = "pending"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None

class SimpleAgent:
    def __init__(self, name:str, role: str = "通用助手", llm: LlmClient = None):
        self.name = name
        self.role = role
        self.state = AgentState.IDLE
        self.memory: AgentMemory = AgentMemory()
        self.tasks: list[Task] = []
        self.tools = {} # 工具
        self.LlmClient = llm
        self.system_prompt = f"""
        你是一个名为{name}的AI Agent，角色是{role}。
        你的能力包括：
        1. 理解和分析用户需求
        2. 制定执行计划
        3. 执行具体任务
        4. 反思和总结经验
        
        你必须始终保持：
        - 逻辑清晰的思考过程
        - 详细的步骤说明
        - 友好和专业的交流方式
        """.strip()
        print(f"🤖 Agent '{name}' 初始化完成")
        print(f"📋 角色: {role}")
        print(f"🔄 状态: {self.state.value}")

    def call_llm(self, messages: list[dict[str, str]], temperature: int = 0) -> str:
        try:
            response = self.LlmClient.chat(messages=messages, temperature=temperature)
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                raise Exception(f"LLM调用失败: {response.status_code}")
        except Exception as e:
            return f"抱歉，在处理您的请求时遇到了问题:{str(e)}"

    def perceive(self,input_data: str) -> dict[str, Any]:
        """感知阶段：理解输入并提取关键信息"""
        print(f"👁️ [{self.name}] 开始感知阶段...")
        self.state = AgentState.PERCEIVING

        message = [
            {"role":"system","content":self.system_prompt},
            {"role":"user","content":f"""
            请分析以下用户输入，提取关键信息：
            
            用户输入: {input_data}
            
            请提供以下分析：
            1. 用户意图是什么？
            2. 需要什么类型的任务？
            3. 有哪些关键参数或条件？
            4. 预期的输出是什么？
            
            请用JSON格式回答，包含以下字段：
            - intent: 用户意图
            - task_type: 任务类型
            - parameters: 关键参数
            - expected_output: 预期输出
            """}
        ]
        response = self.call_llm(message)

        # 尝试解析JSON响应
        try:
            # 提取JSON部分
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                print("✅感知阶段提取到有效的json格式")
                json_str = response[json_start:json_end]
                perception_result = json.loads(json_str)
            else:
                raise ValueError("未找到有效的JSON格式")
        except:
            print("❌感知阶段未提取到有效的json格式")
            # 如果JSON解析失败，使用简化的结构
            perception_result = {
                "intent": "用户查询",
                "task_type": "信息处理",
                "parameters": {"query": input_data},
                "expected_output": "相关回答"
            }
        self.memory.add_short_memory({
            "type":"perceiving",
            "input":input_data,
            "result": perception_result
        })
        print(f"📊 感知结果: {json.dumps(perception_result, ensure_ascii=False, indent=2)}")
        return perception_result

    def plan(self, perception_result: dict[str, Any]) -> list[Task]:
        """规划阶段: 制定执行计划"""
        print(f"📋 [{self.name}] 开始规划阶段...")
        self.state = AgentState.PLANNING
        context = self.memory.get_context()
        message = [
            {"role":"system","content":self.system_prompt},
            {"role": "user", "content": f"""
            基于以下感知结果，请制定详细的执行计划：
    
            感知结果:
            {json.dumps(perception_result, ensure_ascii=False, indent=2)}
    
            上下文信息:
            {context if context else '无'}
    
            请制定执行计划，将任务分解为具体的步骤。
            请用JSON格式回答，包含以下结构：
            {{
              "tasks": [
                {{
                  "id": "task_1",
                  "description": "任务描述",
                  "priority": 1
                }}
              ]
            }}
            """}
        ]
        response = self.call_llm(message)
        # 解析规划结果
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                plan_result = json.loads(json_str)
                tasks_data = plan_result.get('tasks', [])
                print("✅计划阶段提取到有效的json格式")
            else:
                raise ValueError("未找到有效的JSON格式")
        except:
            print("❌计划阶段未提取到有效的json格式")
            # 默认任务
            tasks_data = [{
                "id": "task_1",
                "description": f"处理用户请求: {perception_result.get('intent', '未知')}",
                "priority": 1
            }]

        tasks = []
        for task_data in tasks_data:
            task = Task(
                id=task_data['id'],
                description=task_data['description'],
                priority=task_data.get('priority', 1)
            )
            tasks.append(task)
        self.tasks = tasks
        self.memory.add_short_memory({
            "type":"planning",
            "tasks": [task.description for task in tasks]
        })
        print(f"📝 规划完成，生成 {len(tasks)} 个任务:")
        for i, task in enumerate(tasks, 1):
            print(f"  {i}. {task.description}")

        return tasks

    def execute(self, tasks: list[Task]) -> list[dict[str, Any]]:
        """执行阶段"""
        print(f"⚡ [{self.name}] 开始执行阶段...")
        self.state = AgentState.ACTING

        results = []

        for task in tasks:
            print(f"🔄 执行任务: {task.description}")
            task.status = "in_progress"
            try:
                context = self.memory.get_context()
                messages = [
                    {"role":"system","content":self.system_prompt},
                    {"role": "user", "content": f"""
                    请执行以下任务：
                    
                    任务描述: {task.description}
                    
                    上下文信息:
                    {context if context else '无'}
                    
                    请提供详细的执行结果和过程说明。
                    """}
                ]
                response = self.call_llm(messages)
                # 更新任务状态
                task.status = "completed"
                task.result = response
                task.completed_at = datetime.now().isoformat()

                result = {
                    "task_id": task.id,
                    "task_description": task.description,
                    "status": "success",
                    "result": response
                }

                print(f"✅ 任务完成: {task.description}")
            except Exception as e:
                task.status = "failed"
                task.error = str(e)

                result = {
                    "task_id": task.id,
                    "task_description": task.description,
                    "status": "error",
                    "error": str(e)
                }

                print(f"❌ 任务失败: {task.description} - {str(e)}")

            results.append(result)
            time.sleep(0.5) # 避免api调用过于频繁

        self.memory.add_short_memory({
            "type":"execution",
            "result": results
        })
        return results

    def reflect(self, execution_results: list[dict[str, Any]]) -> str:
        """反思阶段：分析执行结果并总结经验"""
        print(f"🤔 [{self.name}] 开始反思阶段...")
        self.state = AgentState.REFLECTING

        # 统计执行情况
        total_tasks = len(execution_results)
        successful_tasks = len([r for r in execution_results if r['status'] == 'success'])
        failed_tasks = total_tasks - successful_tasks
        # 构建反思提示词
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
        请对以下执行结果进行反思和总结：

        执行统计:
        - 总任务数: {total_tasks}
        - 成功任务: {successful_tasks}
        - 失败任务: {failed_tasks}

        详细结果:
        {json.dumps(execution_results, ensure_ascii=False, indent=2)}

        请提供：
        1. 整体执行效果评价
        2. 成功因素分析
        3. 失败原因分析（如有）
        4. 改进建议
        5. 学到的经验
        """}
        ]
        reflection_result = self.call_llm(messages)
        # 记录反思结果
        self.memory.add_short_memory({
            "type": "reflection",
            "content": reflection_result,
            "stats": {
                "total_tasks": total_tasks,
                "successful_tasks": successful_tasks,
                "failed_tasks": failed_tasks
            }
        })

        # 更新长期记忆
        self.memory.long_memory['last_reflection'] = {
            "timestamp": datetime.now().isoformat(),
            "content": reflection_result,
            "performance": {
                "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0
            }
        }

        self.state = AgentState.IDLE
        print(f"📊 反思完成，成功率: {successful_tasks}/{total_tasks}")

        return reflection_result

    def process(self, user_input: str) -> str:
        """完成的处理逻辑: 感知、规划、执行、反思"""
        print(f"\n🚀 [{self.name}] 开始处理用户请求...")
        print(f"📥 用户输入: {user_input}")
        print("=" * 50)
        try:
            # 1. 感知阶段
            perception_result = self.perceive(user_input)
            print()

            # 2. 规划阶段
            tasks = self.plan(perception_result)
            print()

            # 3. 执行阶段
            execution_results = self.execute(tasks)
            print()

            # 4. 反思阶段
            reflection = self.reflect(execution_results)
            print()

            # 5. 生成最终响应
            successful_results = [r for r in execution_results if r['status'] == 'success']
            if successful_results:
                final_response = "\n\n".join([r['result'] for r in successful_results])
            else:
                final_response = "抱歉，在处理您的请求时遇到了一些问题。请查看详细的执行日志。"

            print("=" * 50)
            print(f"📤 [{self.name}] 处理完成！")
            print(f"💭 最终回答:\n{final_response}")

            return final_response

        except Exception as e:
            self.state = AgentState.ERROR
            error_msg = f"处理过程中出现错误: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

    def get_status(self) -> dict[str, Any]:
        """获取Agent当前状态"""
        return {
            "name": self.name,
            "role": self.role,
            "state": self.state.value,
            "memory_size": len(self.memory.short_memory),
            "tasks_count": len(self.tasks),
            "completed_tasks": len([t for t in self.tasks if t.status == "completed"]),
            "failed_tasks": len([t for t in self.tasks if t.status == "failed"])
        }
if __name__ == '__main__':
    model = os.getenv("LLM_MODEL_ID")
    apiKey = os.getenv("LLM_API_KEY")
    baseUrl = os.getenv("LLM_BASE_URL")
    messages = [
                {"role": "user", "content": "你好，请简单介绍一下你自己。"}
            ]
    llm_client = LlmClient(model= model,apiKey=apiKey, baseUrl=baseUrl)
    #llm_client.chat(messages)
    agent = SimpleAgent(name="小智", role="Agent开发专家", llm = llm_client)
    # 查看Agent初始状态
    print("🔍 Agent初始状态:")
    status = agent.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    # 测试简单问答
   # response1 = agent.process(user_input="请解释一下神恶魔是机器学习，并给出一个简单的例子")

    # 测试复杂任务
    response2 = agent.process(
        "我想要开发一个旅游助手agent，请帮我制定一个构建Agent的流程，这里面我的开发语言为python，并且不涉及模型生成。我将会调用已有的模型来实现，选择的Agent架构模式为ReAct"
    )

    # print("🧠 Agent记忆状况:")
    # print(f"📝 短期记忆条数: {len(agent.memory.short_memory)}")
    # print(f"🗃️ 长期记忆: {list(agent.memory.long_memory.keys())}")
    # print(f"💭 工作记忆: {list(agent.memory.working_memory.keys())}")
    #
    # print("\n📊 最近的记忆内容:")
    # for i, memory in enumerate(agent.memory.short_memory[-3:], 1):
    #     print(f"  {i}. [{memory.get('type', 'unknown')}] {memory.get('timestamp', 'no_time')}")
    #     if memory['type'] == 'reflection':
    #         print(f"     反思内容: {memory.get('content', '')[:100]}...")
    #
    # # 查看任务执行历史
    # print("\n📋 任务执行历史:")
    # for i, task in enumerate(agent.tasks, 1):
    #     print(f"  {i}. [{task.status}] {task.description}")
    #     if task.status == "completed":
    #         print(f"     ✅ 完成时间: {task.completed_at}")
    #     elif task.status == "failed":
    #         print(f"     ❌ 错误信息: {task.error}")