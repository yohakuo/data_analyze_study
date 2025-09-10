# 与数据读取和写入相关的操作,包括从influxdb读取数据,以及将计算好的特征存入clickhouse

import datetime

import clickhouse_connect
from influxdb_client import InfluxDBClient
import pandas as pd

from src import config


def get_all_data() -> pd.DataFrame:
    with InfluxDBClient(
        url=config.INFLUXDB_URL, token=config.INFLUXDB_TOKEN, org=config.INFLUXDB_ORG
    ) as client:
        # 使用 range(start: 0) 获取所有数据
        query = f'''
        from(bucket: "{config.INFLUXDB_BUCKET}")
          |> range(start: 0)
          |> filter(fn: (r) => r["_measurement"] == "{config.MEASUREMENT_NAME}")
          |> filter(fn: (r) => r["_field"] == "{config.FIELD_NAME}")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''

        print("正在从 InfluxDB 查询【所有】数据，请稍候...")
        df = client.query_api().query_data_frame(query=query, org=config.INFLUXDB_ORG)

        if df.empty:
            print("警告：查询结果为空，请检查 InfluxDB 中是否有数据。")
            return df

        print("数据查询成功，正在进行预处理...")

        if config.FIELD_NAME in df.columns and "_time" in df.columns:
            df_cleaned = df[["_time", config.FIELD_NAME]]
            df_cleaned = df_cleaned.set_index("_time")
            df_cleaned.index = pd.to_datetime(df_cleaned.index)
            df_cleaned[config.FIELD_NAME] = pd.to_numeric(
                df_cleaned[config.FIELD_NAME], errors="coerce"
            )
            df_cleaned = df_cleaned.dropna()
            return df_cleaned
        else:
            print("错误：返回的数据中缺少 '_time' 或 '空气湿度' 列。")


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
        url=config.INFLUXDB_URL, token=config.INFLUXDB_TOKEN, org=config.INFLUXDB_ORG
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


def store_features_to_clickhouse(df: pd.DataFrame, table_name: str):
    if df.empty:
        print("\n特征数据为空，跳过存储。")
        return

    print(f"\n开始将特征数据存入 ClickHouse 的表: '{table_name}'...")

    try:
        client = clickhouse_connect.get_client(
            host=config.CLICKHOUSE_HOST,
            port=config.CLICKHOUSE_PORT,
            username="default",
            password="study2025",
        )
        client.command(f"CREATE DATABASE IF NOT EXISTS {config.DATABASE_NAME}")

        # 使用传入的 table_name 参数来构建建表语句
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {config.DATABASE_NAME}.`{table_name}` 
        (
            `时间段` DateTime,
            `分析周期` String,
            `均值` Float64,
            `中位数_Q2` Float64,
            `最大值` Float64,
            `最小值` Float64,
            `Q1` Float64,
            `Q3` Float64,
            `P10` Float64,
            `极差` Float64,
            `超过Q3占时比` Float64,
            `极差的时间变化率` Float64
        )
        ENGINE = MergeTree()
        ORDER BY `时间段`
        """
        client.command(create_table_query)
        print(f"数据库 '{config.DATABASE_NAME}' 和表 '{table_name}' 已准备就绪。")

        # 准备数据用于插入
        # 复制一份数据，避免修改原始的 DataFrame
        df_to_insert = df.copy()
        df_to_insert["分析周期"] = "hourly"
        # 把索引（时间）变回普通列，并重命名以匹配表结构
        df_to_insert = df_to_insert.reset_index()
        df_to_insert.rename(
            columns={
                "_time": "时间段",
                "中位数 (Q2)": "中位数_Q2",
                "10th百分位数": "P10",
            },
            inplace=True,
        )
        # d. 确保列的顺序和类型与表定义一致
        final_columns = [
            "时间段",
            "分析周期",
            "均值",
            "中位数_Q2",
            "最大值",
            "最小值",
            "Q1",
            "Q3",
            "P10",
            "极差",
            "超过Q3占时比",
            "极差的时间变化率",
        ]
        df_to_insert = df_to_insert[final_columns]

        print(f"正在插入 {len(df_to_insert)} 行数据...")
        client.insert_df(f"{config.DATABASE_NAME}.{table_name}", df_to_insert)

        print("数据成功存入 ClickHouse！")

    except Exception as e:
        print(f"存入 ClickHouse 时发生错误: {e}")


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
