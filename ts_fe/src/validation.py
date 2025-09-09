import datetime
import random

import clickhouse_connect
from influxdb_client import InfluxDBClient

from src import config  # 导入中央配置


def get_adjacent_feature_rows(table_name: str):
    """从ClickHouse指定的表中，随机获取【相邻的两行】特征数据。"""
    print(
        f"--- 步骤 1: 正在从 ClickHouse 的表 '{table_name}' 中随机抽取相邻的两条特征数据... ---"
    )
    try:
        client = clickhouse_connect.get_client(
            host=config.CLICKHOUSE_HOST,
            port=config.CLICKHOUSE_PORT,
            username=config.CLICKHOUSE_USER,
            password=config.CLICKHOUSE_PASSWORD,
        )
        count_query = f"SELECT count() FROM {config.DATABASE_NAME}.`{table_name}`"
        total_rows = client.query(count_query).first_row[0]

        if total_rows < 2:
            print("错误：表中的数据不足两行，无法抽取相邻数据。")
            return None

        random_offset = random.randint(0, total_rows - 2)
        query = f"SELECT * FROM {config.DATABASE_NAME}.`{table_name}` ORDER BY `时间段` ASC LIMIT 2 OFFSET {random_offset}"
        adjacent_rows_df = client.query_df(query)
        return adjacent_rows_df
    except Exception as e:
        print(f"连接或查询 ClickHouse 时出错: {e}")
        return None


def get_raw_data_for_hour(start_time_utc: datetime.datetime, field_name: str):
    """根据给定的开始时间和字段名，去 InfluxDB 精准提取这一个小时的原始数据"""
    end_time_utc = start_time_utc + datetime.timedelta(hours=1)

    start_utc_str = start_time_utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    end_utc_str = end_time_utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    print(
        f"\n--- 步骤 2: 正在从 InfluxDB 精准提取 {start_time_utc} (UTC) 的原始数据... ---"
    )

    with InfluxDBClient(
        url=config.INFLUXDB_URL, token=config.INFLUXDB_TOKEN, org=config.INFLUXDB_ORG
    ) as client:
        query = f'''
        from(bucket: "{config.INFLUXDB_BUCKET}")
          |> range(start: {start_utc_str}, stop: {end_utc_str})
          |> filter(fn: (r) => r["_measurement"] == "{config.MEASUREMENT_NAME}")
          |> filter(fn: (r) => r["_field"] == "{field_name}")
          |> keep(columns: ["_value"])
        '''
        result = client.query_api().query(query=query, org=config.INFLUXDB_ORG)
        raw_data_list = [
            record.get_value() for table in result for record in table.records
        ]
        return raw_data_list
