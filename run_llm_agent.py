import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo
import datetime
import json
import time
import sys
import openai
from openai import OpenAI

# 从我们自己的模块中导入所有工具和配置
from src import config
from src.dataset import get_timeseries_data
from src.features import calculate_hourly_features, recalculate_single_hour_features
from src.tools import TOOL_LIST

def find_canonical_field_name(user_term: str) -> str | None:
    """
    根据用户输入的词语，在 config 的别名表中查找并返回规范的字段名。
    """
    # 1. 先检查用户输入的词本身是不是就是一个规范名
    if user_term in config.FIELD_ALIAS_MAP:
        return user_term
    
    # 2. 遍历别名表，查找匹配的别名
    for canonical_name, aliases in config.FIELD_ALIAS_MAP.items():
        if user_term in aliases:
            print(f"  [翻译官]：已将用户输入的 '{user_term}' 翻译为规范字段名 '{canonical_name}'")
            return canonical_name
            
    # 3. 如果找不到，返回 None
    print(f"  [翻译官]：警告！在别名表中找不到与 '{user_term}' 匹配的字段。")
    return None

def create_local_llm_client():
    try:
        client = OpenAI(
            base_url="http://192.168.121.57:11434/v1",
            api_key="ollama"
        )
        print("✅ 成功创建 API 客户端，已指向本地 Ollama 服务。")
        return client
    except Exception as e:
        print(f"❌ 创建 API 客户端失败: {e}")
        sys.exit(1)

def execute_tool_call(tool_call):
    """
    接收 LLM 返回的工具调用指令，并在 Python 中执行对应的函数。
    """
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)
    
    print(f"🤖 LLM 决定调用函数: {function_name}，原始参数: {function_args}")
    
    if function_name == "calculate_hourly_features":
        # a. 从 LLM 的参数中获取用户可能输入的字段名
        user_field_name = function_args.get("field_name")
        
        # b. ===== 关键修改点：让“翻译官”上岗！=====
        canonical_field_name = find_canonical_field_name(user_field_name)
        
        # c. 如果翻译失败，就无法继续
        if not canonical_field_name:
            return {"error": f"无法识别的字段名 '{user_field_name}'。"}
            
        # d. 使用【翻译后】的、规范的字段名，执行后续所有操作
        features_to_calculate = function_args.get("features_to_calculate")
        start_time_str = function_args.get("start_time")
        end_time_str = function_args.get("end_time")

        local_tz = ZoneInfo(config.LOCAL_TIMEZONE)
        start_time_dt = datetime.datetime.strptime(start_time_str, "%Y-%m-%d").replace(tzinfo=local_tz) if start_time_str else None
        end_time_dt = datetime.datetime.strptime(end_time_str, "%Y-%m-%d").replace(tzinfo=local_tz) if end_time_str else None

        raw_df = get_timeseries_data(
            measurement_name=config.INFLUX_MEASUREMENT_NAME,
            field_name=canonical_field_name, # <-- 使用规范名
            start_time=start_time_dt,
            stop_time=end_time_dt
        )

        features_df = calculate_hourly_features(raw_df, canonical_field_name, features_to_calculate) # <-- 使用规范名
        
        return features_df.to_string(max_rows=5, max_cols=20)

    else:
        return {"error": "未找到对应的函数"}

# --- 3. 主函数：代理的交互循环 ---
def main():
    client = create_local_llm_client()
    
    while True:
        user_message = input("\n请提出你的数据分析请求 (输入 'exit' 退出): ")
        if user_message.lower() == 'exit':
            break

        messages = [
            {"role": "system", "content": "你是一个能够处理数据分析任务的工具代理。"},
            {"role": "user", "content": user_message}
        ]

        # 核心步骤：将用户请求和工具箱清单发给 LLM
        response = client.chat.completions.create(
            model="qwen3:8b",
            messages=messages,
            tools=TOOL_LIST, # <-- 这里传递了我们的工具箱清单
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        
        # 检查 LLM 是否决定调用工具
        if response_message.tool_calls:
            tool_call = response_message.tool_calls[0]
            function_response = execute_tool_call(tool_call)
            
            # 将工具的执行结果返回给 LLM
            messages.append(response_message) # 添加 LLM 的工具调用指令
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": function_response
            })
            
            # 再次调用 LLM，让它根据工具的执行结果进行总结
            second_response = client.chat.completions.create(
                model="qwen3:8b",
                messages=messages
            )
            print("\n🤖 模型总结：")
            print(second_response.choices[0].message.content)

        else:
            print("\n🤖 模型回复：")
            print(response_message.content)


if __name__ == "__main__":
    main()