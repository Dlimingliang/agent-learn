import re

def parse_output(text: str):
    """解析LLM的输出，提取Thought和Action。
    """
    # Thought: 匹配到 Action: 或文本末尾
    thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
    # Action: 匹配到文本末尾
    action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else None
    action = action_match.group(1).strip() if action_match else None
    return thought, action

def parse_action(action_text: str):
    """解析Action字符串，提取工具名称和输入。
    """
    match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
    if match:
        return match.group(1), match.group(2)
    return None, None
