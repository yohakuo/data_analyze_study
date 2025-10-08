# 与数据读取和写入相关的操作,包括从influxdb读取数据,以及将计算好的特征存入clickhouse

import datetime
import uuid

import clickhouse_connect
from influxdb_client import InfluxDBClient
import pandas as pd

from src import config
from src.utils import timing_decorator


@timing_decorator
##可指定表，字段和时间
def get_timeseries_data(
    measurement_name: str,
    field_name: str,
    start_time: datetime.datetime = None,
    stop_time: datetime.datetime = None,
) -> pd.DataFrame:
    """
    从 InfluxDB 查询指定表、指定字段、指定时间范围的数据。

    Args:
        measurement_name (str): 要查询的 InfluxDB Measurement (表名)。
        field_name (str): 要查询的字段名。
        start_time (datetime, optional): 查询的开始时间 (带时区)。默认为 None (从头开始)。
        stop_time (datetime, optional): 查询的结束时间 (带时区)。默认为 None (直到现在)。

    Returns:
        pd.DataFrame: 清理好的时间序列数据。
    """

    if start_time is None:
        range_clause = "start: 0"  # 如果没有指定开始时间，则查询所有历史数据
    else:
        # 将 Python 的 datetime 对象转换为 InfluxDB 查询所需的 RFC3339 UTC 字符串
        start_utc_str = start_time.astimezone(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        if stop_time is None:
            range_clause = f"start: {start_utc_str}"
        else:
            stop_utc_str = stop_time.astimezone(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            )
            range_clause = f"start: {start_utc_str}, stop: {stop_utc_str}"

    with InfluxDBClient(
        url=config.INFLUXDB_URL,
        token=config.INFLUXDB_TOKEN,
        org=config.INFLUXDB_ORG,
        timeout=90_000,  # 设置超时为 90,000 毫秒 (90秒)
    ) as client:
        query = f'''
        from(bucket: "{config.INFLUXDB_BUCKET}")
          |> range({range_clause})
          |> filter(fn: (r) => r["_measurement"] == "{measurement_name}")
          |> filter(fn: (r) => r["_field"] == "{field_name}")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''

        print(f"正在从 InfluxDB 的表 '{measurement_name}' 中查询数据...")
        df = client.query_api().query_data_frame(query=query, org=config.INFLUXDB_ORG)

        if df.empty:
            print("警告：在指定的时间范围内，查询结果为空。")
            return df

        print("数据查询成功，正在进行预处理...")

        if field_name in df.columns and "_time" in df.columns:
            df_cleaned = df[["_time", field_name]]
            df_cleaned = df_cleaned.set_index("_time")
            df_cleaned.index = pd.to_datetime(df_cleaned.index)
            df_cleaned[field_name] = pd.to_numeric(df_cleaned[field_name], errors="coerce")
            df_cleaned = df_cleaned.dropna()
            return df_cleaned
        else:
            print(f"错误：返回的数据中缺少 '_time' 或 '{field_name}' 列。")
            return pd.DataFrame()


@timing_decorator
def store_features_to_clickhouse(
    df: pd.DataFrame,
    table_name: str,
    field_name: str,
    device_id: str,
    temple_id: str,
    stats_cycle: str,
):
    """
    接收【宽格式】的特征 DataFrame，为其生成唯一ID，转换为【长格式】，并存入 ClickHouse。
    """
    if df.empty:
        print("\n特征数据为空，跳过存储。")
        return

    print(f"\n开始处理并存入 ClickHouse 的表: '{table_name}'...")

    try:
        df_long = df.reset_index()
        feature_columns = [col for col in df.columns if col not in ["分析周期"]]
        df_long = pd.melt(
            df_long,
            id_vars=["_time"],
            value_vars=feature_columns,
            var_name="feature_key",
            value_name="feature_value",
        )

        df_long["stat_id"] = [uuid.uuid4() for _ in range(len(df_long))]
        df_long["temple_id"] = temple_id
        df_long["device_id"] = device_id
        df_long["monitored_variable"] = field_name
        df_long["stats_cycle"] = stats_cycle
        df_long["created_at"] = datetime.datetime.now()
        df_long.rename(columns={"_time": "stats_start_time"}, inplace=True)

        final_columns = [
            "stat_id",
            "temple_id",
            "device_id",
            "stats_start_time",
            "monitored_variable",
            "stats_cycle",
            "feature_key",
            "feature_value",
            "created_at",
        ]
        df_to_insert = df_long[final_columns]
        df_to_insert["feature_value"] = df_to_insert["feature_value"].astype(str)
        df_to_insert["standby_field01"] = ""

        client = clickhouse_connect.get_client(
            host=config.CLICKHOUSE_HOST,
            port=config.CLICKHOUSE_PORT,
            username=config.CLICKHOUSE_USER,
            password=config.CLICKHOUSE_PASSWORD,
        )
        client.command(f"CREATE DATABASE IF NOT EXISTS {config.DATABASE_NAME}")
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {config.DATABASE_NAME}.`{table_name}`
        (
            `stat_id` UUID,
            `temple_id` CHAR(20),
            `device_id` CHAR(30),
            `stats_start_time` DATETIME,
            `monitored_variable` CHAR(20),
            `stats_cycle` CHAR(10),
            `feature_key` CHAR(30),
            `feature_value` VARCHAR(30),
            `standby_field01` CHAR(30),
            `created_at` TIMESTAMP
        ) ENGINE = MergeTree() ORDER BY (stats_start_time, device_id, feature_key)
        """
        client.command(create_table_query)

        print(f"  ...正在插入 {len(df_to_insert)} 行长格式特征数据...")
        client.insert_df(f"{config.DATABASE_NAME}.`{table_name}`", df_to_insert)

    except Exception as e:
        print(f"❌ 存入 ClickHouse 时发生错误: {e}")


def get_latest_timestamp_from_clickhouse() -> pd.Timestamp:
    """查询 ClickHouse，获取已有特征数据的最新时间戳。"""
    try:
        client = clickhouse_connect.get_client(
            host=config.CLICKHOUSE_HOST,
            port=config.CLICKHOUSE_PORT,
            username=config.CLICKHOUSE_USER,
            password=config.CLICKHOUSE_PASSWORD,
        )
        tables = client.query_df(f"SHOW TABLES FROM {config.DATABASE_NAME}")
        if config.TABLE_NAME not in tables["name"].values:
            return None

        query = f"SELECT max(`时间段`) FROM {config.DATABASE_NAME}.{config.TABLE_NAME}"
        result = client.query(query)
        latest_time = result.first_row[0] if result.first_row else None

        if latest_time:
            return pd.to_datetime(latest_time).tz_localize("UTC")
        return None
    except Exception as e:
        print(f"查询 ClickHouse 最新时间戳时出错: {e}")
        return None
