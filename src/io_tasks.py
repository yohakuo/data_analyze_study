# 包含所有“高级”的数据I/O任务。
import datetime

from clickhouse_driver import Client
import pandas as pd

# ‼️ 导入你【已有的】底层函数
from src.io import store_dataframe_to_clickhouse

# --- [E] 提取模块 ---


def get_distinct_ids(client: Client, db: str, table: str, id_column: str = "device_id") -> list:
    """
    [E] 从源表获取所有唯一的 ID 列表。
    这是管线(Pipeline)的第一步。
    """
    print(f"   ► [E] 正在从 {db}.{table} 查询所有唯一的 '{id_column}'...")
    query = f"SELECT DISTINCT `{id_column}` FROM `{db}`.`{table}`"
    try:
        result = client.execute(query)
        id_list = [row[0] for row in result if row and row[0] is not None]
        print(f"   ► [E] 找到了 {len(id_list)} 个唯一的 ID。")
        return id_list
    except Exception as e:
        print(f"❌ [E] 查询 distinct ID 失败: {e}")
        return []


def get_data_for_id(
    client: Client,
    db: str,
    table: str,
    device_id: str,
    id_column: str = "device_id",
    time_column: str = "time",
) -> pd.DataFrame:
    """
    [E] 提取单个 ID 的【所有】历史数据。
    这是管线(Pipeline)循环中的提取步骤。
    """
    # print(f"   ► [E] 正在提取 '{device_id}' 的所有数据...")
    full_table_path = f"`{db}`.`{table}`"
    params = {"id": device_id}

    query = f"""
        SELECT * FROM {full_table_path} 
        WHERE `{id_column}` = %(id)s 
        ORDER BY `{time_column}`
    """
    try:
        df = client.query_dataframe(query, params=params)

        # 你的 dataset.py 在提取后会转换时间
        df[time_column] = pd.to_datetime(df[time_column], utc=True)

        return df
    except Exception as e:
        print(f"❌ [E] 提取 ID '{device_id}' 数据失败: {e}")
        return pd.DataFrame()


# --- [L] 加载模块 ---


def load_features_to_clickhouse(
    features_df: pd.DataFrame,
    client: Client,
    db: str,
    table: str,
    # (元数据)
    temple_id: str,  # 假设
    stats_cycle: str,
):
    """
    [L] 加载包装器：
    1. 为特征数据添加【批次】元数据。
    2. 调用你【已有的】 store_dataframe_to_clickhouse 函数。
    """
    if features_df.empty:
        print("   ► [L] 特征数据为空，跳过存储。")
        return

    # 1. 添加元数据
    # (注意： 'device_id', 'time', 'field_name', 'feature_key', 'value'
    #  应该已在 Transform 步骤中生成)
    features_df["temple_id"] = temple_id
    features_df["stats_cycle"] = stats_cycle

    # 2. 调用真正的存储函数
    try:
        # ‼️ 这是对你 dataset.py 中函数的【复用】
        store_dataframe_to_clickhouse(
            df=features_df,
            client=client,
            database_name=db,
            table_name=table,
        )
        # print(f"   ► [L] 成功调用存储函数存入 {len(features_df)} 行。")
    except Exception as e:
        print(f"❌ [L] 存储失败: {e}")
