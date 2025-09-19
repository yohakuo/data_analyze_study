import openai

# --- 1. 配置本地 LLM 服务器 ---
# 服务器地址
LOCAL_BASE_URL = "http://192.168.121.57:11434/v1" 
# API 密钥对于本地 Ollama 服务来说不是必需的，但 openai 库要求必须提供一个值
LOCAL_API_KEY = "ollama" 
# 模型名称
MODEL_NAME = "qwen3:8b"


# --- 2. 创建一个 OpenAI 客户端，但让它指向我们的本地服务器 ---
try:
    client = openai.OpenAI(
        base_url=LOCAL_BASE_URL,
        api_key=LOCAL_API_KEY,
    )
    print("✅ 成功创建 API 客户端，已指向本地服务器。")
except Exception as e:
    print(f"❌ 创建 API 客户端失败: {e}")
    exit()


def list_available_models():
    """获取并打印服务器上所有可用的模型列表。"""
    print("\n--- 步骤 1: 正在获取可用模型列表... ---")
    try:
        models = client.models.list()
        print("✅ 成功获取！服务器上的可用模型有：")
        for model in models.data:
            print(f"  - {model.id}")
    except Exception as e:
        print(f"❌ 获取模型列表失败: {e}")


def chat_with_llm():
    """向指定的 LLM 发送消息并获取回复。"""
    print(f"\n--- 步骤 2: 正在向模型 '{MODEL_NAME}' 发送消息... ---")
    try:
        # 这是标准的 OpenAI 对话 API 调用
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个乐于助人的AI助手。"},
                {"role": "user", "content": "你好！请用中文简单介绍一下你自己。"}
            ]
        )
        
        # 从回复中提取并打印出模型的回答
        ai_message = response.choices[0].message.content
        print("\n🤖 模型回复：")
        print(ai_message)
        
    except Exception as e:
        print(f"❌ 与模型对话失败: {e}")


if __name__ == "__main__":
    list_available_models()
    chat_with_llm()