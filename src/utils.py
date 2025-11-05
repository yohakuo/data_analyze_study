from datetime import datetime
from functools import wraps
import random
import re
import time

import pandas as pd
import pandas.api.types as ptypes

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


def generate_create_table_sql(
    df: pd.DataFrame, full_table_path: str, engine: str = "MergeTree", order_by: str = None
) -> str:
    """
    [“自动建表”工具]
    从 Pandas DataFrame 自动推断数据类型，并生成 ClickHouse 的 CREATE TABLE 语句。
    """
    # --- 类型映射字典 ---
    # 将 Pandas/Numpy 的 dtypes 映射为 ClickHouse 的数据类型
    type_mapping = {
        "int8": "Int8",
        "int16": "Int16",
        "int32": "Int32",
        "int64": "Int64",
        "uint8": "UInt8",
        "uint16": "UInt16",
        "uint32": "UInt32",
        "uint64": "UInt64",
        "float32": "Float32",
        "float64": "Float64",
        "object": "String",  # 默认将 object 视为 String
        "bool": "UInt8",
    }

    column_definitions = []

    # 遍历所有列，包括被重置的索引
    for col, dtype in df.dtypes.items():
        # 清理列名，确保它们是合法的SQL标识符
        col_name = f"`{re.sub(r'W+', '_', col).strip('_')}`"

        # 特殊处理：日期时间类型
        if ptypes.is_datetime64_any_dtype(dtype):
            col_type = "DateTime"  # 使用 clickhouse_driver 兼容的 DateTime
        # 特殊处理：字符串/对象类型
        elif ptypes.is_object_dtype(dtype):
            col_type = "String"
        # 其他数值/布尔类型
        else:
            col_type = type_mapping.get(dtype.name, "String")  # 默认为 String

        column_definitions.append(f"{col_name} {col_type}")

    # --- 拼接 SQL 语句 ---
    columns_sql = ",\n        ".join(column_definitions)

    if not order_by and df.columns.any():
        # 如果未指定排序键，默认使用第一列
        order_by_col = f"`{re.sub(r'W+', '_', df.columns[0]).strip('_')}`"
        order_by = f"({order_by_col})"

    # 我们直接在这里生成完整的 SQL，并包含 IF NOT EXISTS
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {full_table_path}
    (
        {columns_sql}
    )
    ENGINE = {engine}()
    ORDER BY {order_by}
    """
    return create_sql
