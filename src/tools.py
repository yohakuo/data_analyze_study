# src/tools.py

#  calculate_hourly_features 
#  name必须与 Python 函数的名字完全一致
#  description 是 LLM “阅读”和“理解”的部分。描述越清晰，LLM 犯错的可能性越小。
# 定义了函数需要的参数，包括它们的类型、作用描述和是否必需。
TOOL_CALCULATE_HOURLY_FEATURES = {
    "type": "function",
    "function": {
        "name": "calculate_hourly_features",
        "description": "计算某一段时间序列数据在每个小时内的统计特征，包括均值（Q2）、最大值、最小值、标准差、Q1、Q3、超过Q3占时比、前百分之10（P10）。"
        "注意：如果用户说'湿度'或'相对湿度'，请使用规范名称 '空气湿度（%）'",
        "parameters": {
            "type": "object",
            "properties": {
                "field_name": {
                    "type": "string",
                    "description": "需要计算特征的字段名称，例如 '空气湿度' 或 '空气温度'。",
                },
                "features_to_calculate": {
                    "type": "array",
                    "description": "要计算的特征列表。例如 ['均值', '最大值']。",
                    "items": {"type": "string"}
                },
                "start_time": {
                    "type": "string",
                    "description": "查询的起始日期，格式为 YYYY-MM-DD。"
                },
                "end_time": {
                    "type": "string",
                    "description": "查询的结束日期，格式为 YYYY-MM-DD。"
                }
            },
            "required": ["field_name", "features_to_calculate"]
        }
    }
}

# 存放多个工具的定义
TOOL_LIST = [
    TOOL_CALCULATE_HOURLY_FEATURES,
]