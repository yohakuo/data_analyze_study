import pytest
import pandas as pd
import numpy as np
import clickhouse_connect
from src import config
from src.dataset import get_timeseries_data
from src.features import recalculate_single_hour_features

# --- 测试配置 ---
TABLE_TO_VALIDATE = 'features_caculate' 
FIELD_TO_VALIDATE = '空气湿度（%）'

# --- 测试辅助函数 ---
def get_features_for_random_hour(client):
    """从ClickHouse随机选择一个小时，并获取该小时的所有特征。"""
    query = f"""
    SELECT * FROM {config.DATABASE_NAME}.`{TABLE_TO_VALIDATE}`
    WHERE `stats_start_time` = (
        SELECT `stats_start_time`
        FROM {config.DATABASE_NAME}.`{TABLE_TO_VALIDATE}`
        ORDER BY rand()
        LIMIT 1
    )
    """
    df = client.query_df(query)
    # 将 feature_key 设置为索引，方便后面快速查找
    return df.set_index('feature_key')

# --- 核心测试函数 ---
def test_feature_calculation_correctness():
    """
    执行完整的端到端验证：
    1. 从 ClickHouse 获取一个随机小时的特征作为“标准答案”。
    2. 从 InfluxDB 获取对应的原始数据。
    3. 重新计算特征。
    4. 对比两个结果。
    """
    print("\n--- 开始端到端特征验证 ---")
    
    # --- 准备阶段 ---
    # 1. 连接 ClickHouse 并获取“标准答案”
    ch_client = clickhouse_connect.get_client(
        host=config.CLICKHOUSE_HOST, port=config.CLICKHOUSE_PORT,
        username=config.CLICKHOUSE_USER, password=config.CLICKHOUSE_PASSWORD
    )
    stored_features_df = get_features_for_random_hour(ch_client)
    
    if stored_features_df.empty:
        pytest.fail("无法从 ClickHouse 获取测试样本，测试终止。")

    # 从标准答案中获取时间戳
    timestamp_utc = pd.to_datetime(stored_features_df['stats_start_time'].iloc[0]).tz_localize('UTC')
    print(f"正在验证时间点: {timestamp_utc}")

    # 2. 从 InfluxDB 获取原始数据
    raw_df = get_timeseries_data(
        measurement_name=config.MEASUREMENT_NAME, # 原始数据表名
        start_time=timestamp_utc,
        stop_time=timestamp_utc + pd.Timedelta(hours=1),
        field_name=FIELD_TO_VALIDATE
    )
    
    if raw_df.empty:
        pytest.skip(f"在 {timestamp_utc} 这个小时没有找到原始数据，跳过测试。")

    raw_data_series = raw_df[FIELD_TO_VALIDATE]

    # --- 计算与验证阶段 ---
    # 3. 重新计算特征
    recalculated_features = recalculate_single_hour_features(raw_data_series)
    
    print("\n--- 开始逐项对比 ---")
    # 4. 逐一对比每一个特征
    for feature_key, recalculated_value in recalculated_features.items():
        # 从 ClickHouse 的结果中，根据 feature_key 找到对应的行
        if feature_key in stored_features_df.index:
            stored_value_str = stored_features_df.loc[feature_key, 'feature_value']
            # 将存储的字符串值转换回浮点数进行比较
            stored_value = float(stored_value_str)
            
            print(f"正在对比特征: {feature_key}")
            
            # 使用 assert 进行断言
            assert np.isclose(recalculated_value, stored_value, atol=1e-5), \
                f"特征 '{feature_key}' 验证失败！计算值: {recalculated_value}, 存储值: {stored_value}"
        else:
            print(f"警告：计算出的特征 '{feature_key}' 在 ClickHouse 样本中未找到。")