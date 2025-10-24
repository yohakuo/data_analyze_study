from datetime import datetime
from functools import wraps
import random
import re
import time

import pandas as pd

from src import config  # 导入中央配置


def timing_decorator(func):
    """
    打印函数执行时间。
    """

    # @wraps(func) 能帮助保留原函数的名称等信息
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()

        # *args, **kwargs 是魔法语法，它能让“助理”服务于任何类型的函数
        # 无论有多少个参数
        result = func(*args, **kwargs)

        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"✅  {func.__name__} 执行完毕，耗时: {duration:.2f} 秒")

        return result

    return wrapper


def process_dataframe(df_raw, column_map):
    """
    执行基础的数据转换。
    1. 根据 column_map 重命名字段
    2. 筛选出需要的字段
    3. 添加 'created_at' 字段
    4. 转换通用的 'time' 字段
    """
    if df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()

    # 重命名字段 (column_map 定义)
    df.rename(columns=column_map, inplace=True)
    expected_cols = list(column_map.values())
    # 确保 Excel 里的列都存在
    valid_cols = [col for col in expected_cols if col in df.columns]
    missing_cols = set(expected_cols) - set(valid_cols)
    if missing_cols:
        print(f"警告: 映射中定义了 {missing_cols} 列, 但在Excel中找不到，将被忽略。")

    df = df[valid_cols]

    df["created_at"] = pd.to_datetime(datetime.now())
    # 换通用的 'time' 字段
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df.dropna(subset=["time"], inplace=True)
    return df
