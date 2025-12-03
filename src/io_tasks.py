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

    query = f"SELECT DISTINCT `{id_column}` FROM `{db}`.`{table}`"
    try:
        result = client.execute(query)
        id_list = [row[0] for row in result if row and row[0] is not None]
        # print(f"   ► [E] 找到了 {len(id_list)} 个唯一的 ID。")
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
    stats_cycle: str,
):
    """
    [L] 加载包装器：
    1. 添加元数据。
    2. 重命名列以匹配目标表（图一）。
    3. 调用 store_dataframe_to_clickhouse。
    """
    if features_df.empty:
        print("   ► [L] 特征数据为空，跳过存储。")
        return

    # 复制一份以避免 SettingWithCopyWarning
    df_to_load = features_df.copy()

    # 1. 添加元数据
    df_to_load["stats_cycle"] = stats_cycle
    df_to_load["created_at"] = datetime.datetime.now(datetime.timezone.utc)

    # 2. 定义列名映射
    COLUMN_MAP = {
        "time": "stats_start_time",
        "field_name": "monitored_variable",
        "value": "feature_value",
    }

    df_to_load.rename(columns=COLUMN_MAP, inplace=True)

    # 3. 确保数据类型匹配目标表
    df_to_load["feature_value"] = df_to_load["feature_value"].astype(str)

    # 4. (可选) 确保只包含目标表中的列
    #    (注意: stat_id 和 standby_field01 应由数据库处理)
    FINAL_COLUMNS = [
        "temple_id",
        "device_id",
        "stats_start_time",
        "monitored_variable",
        "stats_cycle",
        "feature_key",
        "feature_value",
        "created_at",
    ]

    # 筛选出最终的列
    final_df = df_to_load[FINAL_COLUMNS]

    try:
        store_dataframe_to_clickhouse(
            df=final_df,  # 传入重命名和清理后的 DataFrame
            client=client,
            database_name=db,
            table_name=table,
        )
    except Exception as e:
        print(f"❌ [L] 存储失败: {e}")
