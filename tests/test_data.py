import clickhouse_connect
import numpy as np
import pandas as pd
import pytest
from src import config
from src.dataset import get_timeseries_data
from src.features import recalculate_single_hour_features

# --- 测试配置 ---
NUM_SAMPLES_TO_TEST = 100  # 我们要随机抽查100行
TABLE_TO_VALIDATE = "humidity_hourly_features"  # 你要验证的 ClickHouse 表
FIELD_TO_VALIDATE = "空气湿度"  # 对应的原始数据字段


# --- 测试辅助函数：从 ClickHouse 获取随机样本 ---
def get_random_samples_from_clickhouse(num_samples):
    """从 ClickHouse 随机获取指定数量的样本行。"""
    print(f"\n正在从 ClickHouse 的表 '{TABLE_TO_VALIDATE}' 中随机抽取 {num_samples} 个样本...")
    try:
        client = clickhouse_connect.get_client(
            host=config.CLICKHOUSE_HOST,
            port=config.CLICKHOUSE_PORT,
            username=config.CLICKHOUSE_USER,
            password=config.CLICKHOUSE_PASSWORD,
        )
        # 使用 ORDER BY rand() LIMIT N 来高效地实现随机抽样
        query = f"SELECT * FROM {config.DATABASE_NAME}.`{TABLE_TO_VALIDATE}` ORDER BY rand() LIMIT {num_samples}"
        samples_df = client.query_df(query)

        if len(samples_df) < num_samples:
            print(f"警告：表中数据行数 ({len(samples_df)}) 少于期望的样本数 ({num_samples})。")

        # 将 DataFrame 转换为一个字典列表，方便 pytest 参数化
        return samples_df.to_dict("records")
    except Exception as e:
        print(f"连接或查询 ClickHouse 时出错: {e}")
        return []


# --- 核心测试：使用 pytest 参数化来为每个样本创建一个独立的测试用例 ---

# 1. 在测试运行前，先准备好所有的测试样本
test_samples = get_random_samples_from_clickhouse(NUM_SAMPLES_TO_TEST)


@pytest.mark.parametrize("stored_features_row", test_samples)
def test_feature_calculation_correctness(stored_features_row):
    """
    这是一个参数化的测试函数。
    Pytest 会为 test_samples 列表中的每一个样本，都独立地运行一次这个函数。
    """
    # --- 准备阶段 ---
    # a. 从样本中获取时间戳
    timestamp_utc = pd.to_datetime(stored_features_row["时间段"]).tz_localize("UTC")

    # b. 根据时间戳，去 InfluxDB 精准提取这一个小时的原始数据
    #    (注意：这里我们复用了之前项目里的 get_timeseries_data 函数)
    raw_df = get_timeseries_data(
        measurement_name=config.MEASUREMENT_NAME,  # 假设原始数据表名在 config 里
        field_name=FIELD_TO_VALIDATE,
        start_time=timestamp_utc,
        stop_time=timestamp_utc + pd.Timedelta(hours=1),
    )

    # 如果某个小时没有原始数据，我们跳过这个测试
    if raw_df.empty:
        pytest.skip(f"在 {timestamp_utc} 这个小时没有找到原始数据，跳过测试。")

    raw_data_series = raw_df[FIELD_TO_VALIDATE]

    # --- 计算阶段 ---
    # c. 使用我们 src/features.py 里的“设计蓝图”，重新计算特征
    recalculated_features = recalculate_single_hour_features(raw_data_series)

    # --- 验证阶段 (核心) ---
    # d. 逐一对比每一个特征
    for feature_name, recalculated_value in recalculated_features.items():
        # 从 ClickHouse 样本中获取存储的值
        stored_value = stored_features_row[feature_name]

        # 使用 assert 和 np.isclose 进行断言
        # 如果两者不接近，pytest 会立刻报错并停止这个测试用例
        assert np.isclose(recalculated_value, stored_value, atol=1e-5), (
            f"特征 '{feature_name}' 在时间点 {timestamp_utc} 验证失败！"
            f" 计算值: {recalculated_value}, 存储值: {stored_value}"
        )
