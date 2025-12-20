import json
import logging

import pandas as pd
import streamlit as st
import yaml

from src.features.calculator import FeatureCalculator
from src.features.llm import HybridLLMService
from src.io import load_heritage_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="监测数据管理", layout="wide")

# --- 侧边栏：配置 ---
st.sidebar.title("智能体配置")
model_source = st.sidebar.radio("选择模型来源", ["Local (本地 Ollama)", "Online (OpenAI Compatible)"])

if model_source == "Local (本地 Ollama)":
    mode = "local"
    base_url = st.sidebar.text_input("Local API URL", "http://localhost:11434/v1")
    api_key = "ollama"
    model_name = st.sidebar.text_input("模型名称", "qwen2.5:7b")
else:
    mode = "online"
    base_url = st.sidebar.text_input(
        "Online API URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    api_key = st.sidebar.text_input("API Key", type="password")
    model_name = st.sidebar.text_input("模型名称", "qwen-plus")


# --- 加载资源 ---
@st.cache_resource
def get_calculator():
    return FeatureCalculator()


@st.cache_data
def get_data():
    try:
        return load_heritage_data("data")  # 读取 data/ 文件夹
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()


# 初始化服务
try:
    calculator = get_calculator()
    df = get_data()
    # Initialize LLM service with current config
    # Note: We don't cache this resource to allow dynamic config changes, 
    # but in production you might want to cache if initialization is heavy.
    llm_service = HybridLLMService(
        mode=mode, 
        base_url=base_url, 
        api_key=api_key, 
        model_name=model_name
    )
except Exception as e:
    st.error(f"初始化失败: {e}")
    st.stop()

if df.empty:
    st.warning("data/ 目录下没有找到 CSV 数据，请放入数据后刷新。")
    st.stop()


# --- 业务逻辑 ---
def execute_analysis(calculator, df, metric_def, filter_params):
    """
    根据指标定义和过滤参数执行分析
    """
    # 1. 动态过滤
    filtered_df = df.copy()

    # 筛选点位 (示例 logic)
    if "site_id" in filter_params and filter_params["site_id"]:
        if "site_id" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["site_id"] == filter_params["site_id"]]

    # 筛选时间
    try:
        if "start_date" in filter_params and filter_params["start_date"]:
             filtered_df = filtered_df[filtered_df.index >= filter_params["start_date"]]
        
        if "end_date" in filter_params and filter_params["end_date"]:
             filtered_df = filtered_df[filtered_df.index <= filter_params["end_date"]]
    except Exception as e:
        logger.warning(f"Error filtering by date: {e}")

    if filtered_df.empty:
        return None, "过滤后无数据"

    # 2. 准备计算参数
    field = metric_def["data_field"]
    features = metric_def["calculation_logic"]
    # 优先用LLM提取的频率，否则用默认
    freq = filter_params.get("freq") or metric_def["default_freq"]

    # 3. 执行计算
    try:
        result_df = calculator.calculate_statistical_features(
            filtered_df, field_name=field, feature_list=features, freq=freq
        )
        return result_df, None
    except Exception as e:
        return None, f"计算错误: {e}"


# --- 界面交互区域 ---
st.title(" 环境监测语义分析系统")
st.caption(f"当前接入数据：{len(df)} 条 | 时间范围：{df.index.min()} 至 {df.index.max()}")

# 聊天输入
user_query = st.chat_input(
    "请输入指令，例如：'帮我看看温度的变化趋势' 或 '分析一下湿度的每日统计'"
)

if user_query:
    # 4.1 显示用户提问
    with st.chat_message("user"):
        st.write(user_query)

    # 4.2 LLM 语义分析
    with st.chat_message("assistant"):
        with st.spinner("AI 正在思考..."):
            intent = llm_service.parse_intent(user_query)
        
        if "error" in intent:
             st.error(f"意图识别失败: {intent['error']}")
             st.stop()
             
        metric_id = intent.get("metric_id")
        params = intent.get("params", {})
        
        if metric_id:
            # 查找指标定义
            metric_config = next(
                (m for m in llm_service.metrics_def["metrics"] if m["id"] == metric_id), 
                None
            )
            
            if metric_config:
                st.success(f"已识别意图：**{metric_config['name']}**")
                with st.expander("查看解析参数"):
                    st.json(params)
                
                # 执行分析
                result_df, error_msg = execute_analysis(calculator, df, metric_config, params)
                
                if error_msg:
                    st.warning(error_msg)
                else:
                    st.subheader("📊 分析图表")
                    
                    # 简单处理 MultiIndex 列名以便绘图
                    if isinstance(result_df.columns, pd.MultiIndex):
                        result_df.columns = [
                            "_".join(col).strip() for col in result_df.columns.values
                        ]
                    
                    viz_type = metric_config.get("viz_type", "line")
                    if viz_type == "line":
                        st.line_chart(result_df)
                    elif viz_type == "area":
                        st.area_chart(result_df)
                    elif viz_type == "bar":
                        st.bar_chart(result_df)
                    else:
                        st.line_chart(result_df)

                    with st.expander("查看详细数据"):
                        st.dataframe(result_df)
            else:
                st.error(f"未找到指标定义: {metric_id}")
        else:
            st.warning("抱歉，我没有理解您的意图，或者该分析尚不支持。")
