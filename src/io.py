"""
与数据库连接、数据读写、预处理相关的核心函数。
"""

import csv
import datetime
import os
import re
from zoneinfo import ZoneInfo

from clickhouse_driver import Client
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import numpy as np
import pandas as pd

from src import config, table_definitions
from src.utils import generate_create_table_sql, timing_decorator

# ====================================================================
#   数据库连接 (Database Clients)
# ====================================================================


def get_clickhouse_client(target: str = "shared") -> Client:
    """
    根据目标名称 ('local' 或 'shared')，创建并返回一个 ClickHouse 客户端。
    这个客户端【没有】连接到任何特定的数据库。
    """
    print(f"  ► 正在创建指向 '{target}' ClickHouse 服务器的连接...")
    try:
        if target == "local":
            return Client(
                host=config.CLICKHOUSE_HOST,
                port=config.CLICKHOUSE_PORT,
                user=config.CLICKHOUSE_USER,
                password=config.CLICKHOUSE_PASSWORD,
            )
        elif target == "shared":
            return Client(
                host=config.CLICKHOUSE_SHARED_HOST,
                port=config.CLICKHOUSE_SHARED_PORT,
                user=config.CLICKHOUSE_SHARED_USER,
                password=config.CLICKHOUSE_SHARED_PASSWORD,
            )
        else:
            raise ValueError(f"未知的数据库目标: '{target}'。请选择 'local' 或 'shared'。")
    except Exception as e:
        print(f"❌ 连接到 '{target}' ClickHouse 时出错: {e}")
        raise


# ====================================================================
#   数据读写 (Data I/O)
# ====================================================================


def read_excel_data(file_path: str, sheet_name=0, header_row=0, dtypes=None) -> pd.DataFrame:
    """
    通用 Excel 读取函数。

    Args:
        file_path (str): 文件路径。
        sheet_name (int or str): 表单名 (默认为第一个)。
        header_row (int): 标题行 (0 是第一行)。
        dtypes (dict, optional): 指定列的数据类型。

    Returns:
        pd.DataFrame: 清理后的数据帧，如果失败则返回空的 DataFrame。
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row, dtype=dtypes)
        df_cleaned = df.dropna(how="all")
        return df_cleaned
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
    except Exception as e:
        print(f"错误: 读取 Excel 文件 {file_path} 失败: {e}")
    return pd.DataFrame()


@timing_decorator
def get_timeseries_data(
    measurement_name: str,
    field_name: str,
    start_time: datetime.datetime = None,
    stop_time: datetime.datetime = None,
) -> pd.DataFrame:
    """
    从 InfluxDB 查询指定表、字段和时间范围的时间序列数据。
    """
    range_clause = "start: 0"
    if start_time:
        start_utc_str = start_time.astimezone(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        range_clause = f"start: {start_utc_str}"
        if stop_time:
            stop_utc_str = stop_time.astimezone(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            )
            range_clause += f", stop: {stop_utc_str}"

    with InfluxDBClient(
        url=config.INFLUXDB_URL,
        token=config.INFLUXDB_TOKEN,
        org=config.INFLUXDB_ORG,
        timeout=90_000,  # 90秒超时
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
            df_cleaned = df[["_time", field_name]].set_index("_time")
            df_cleaned.index = pd.to_datetime(df_cleaned.index)
            df_cleaned[field_name] = pd.to_numeric(df_cleaned[field_name], errors="coerce")
            df_cleaned.dropna(inplace=True)
            return df_cleaned
        else:
            print(f"错误：返回的数据中缺少 '_time' 或 '{field_name}' 列。")
            return pd.DataFrame()


def create_table_if_not_exists(
    client: Client, db_name: str, table_name: str, schema_template: str
) -> bool:
    """
    根据提供的模板，创建数据库和表（如果不存在）。

    Args:
        client: ClickHouse 数据库连接。
        db_name: 数据库名。
        table_name: 要创建的表名。
        schema_template: 包含 {db_name} 和 {table_name} 占位符的 SQL 字符串。

    Returns:
        bool: 成功返回 True，失败返回 False。
    """
    try:
        client.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        create_table_query = schema_template.format(db_name=db_name, table_name=table_name)
        client.execute(create_table_query)
        # print(f"表 {db_name}.`{table_name}` 检查/创建 成功。")
        return True
    except Exception as e:
        print(f"错误: 创建表 {db_name}.`{table_name}` 失败: {e}")
        return False


def _create_raw_tables_if_not_exists(client: Client):
    """
    使用 src.table_definitions 中的模板创建原始数据表（如果不存在）。
    """
    db_name = config.CLICKHOUSE_SHARED_DB

    # 1. 从 config 获取表名，从 table_definitions 获取表结构
    table_temp_name = config.RAW_SENSOR_MAPPING_CONFIG["无线温湿度传感器"]["clickhouse_table"]
    create_table_if_not_exists(
        client, db_name, table_temp_name, table_definitions.RAW_TEMP_HUMIDITY_SCHEMA
    )

    # 2. 同上，处理 CO2 表
    table_co2_name = config.RAW_SENSOR_MAPPING_CONFIG["无线二氧化碳传感器"]["clickhouse_table"]
    create_table_if_not_exists(client, db_name, table_co2_name, table_definitions.RAW_CO2_SCHEMA)


def get_global_time_range(
    client: Client, database_name: str, table_name: str, time_column: str = "time"
) -> tuple[datetime.datetime, datetime.datetime]:
    """
    查询 ClickHouse 表，找出数据中最早和最晚的时间戳。
    """
    full_table_path = f"`{database_name}`.`{table_name}`"
    query = f"SELECT min(`{time_column}`), max(`{time_column}`) FROM {full_table_path}"

    try:
        result = client.execute(query)
        if not result or not result[0]:
            raise Exception("表为空或无法查询到时间范围。")

        min_date, max_date = result[0]

        # 确保返回的是带时区的 datetime 对象
        min_date = pd.to_datetime(min_date).tz_localize("UTC")
        max_date = pd.to_datetime(max_date).tz_localize("UTC")

        return min_date, max_date

    except Exception as e:
        print(f"❌  侦察时间范围时出错: {e}")
        raise


def import_csv_to_influx(csv_filepath: str):
    """
    读取指定的 CSV 文件，并将其内容批量导入到 InfluxDB。
    """
    local_tz = ZoneInfo(config.LOCAL_TIMEZONE)
    batch_size = 5000

    with InfluxDBClient(
        url=config.INFLUXDB_URL, token=config.INFLUXDB_TOKEN, org=config.INFLUXDB_ORG
    ) as client:
        write_api = client.write_api(write_options=SYNCHRONOUS)
        points_buffer = []
        total_count = 0

        try:
            with open(csv_filepath, "r", encoding="gbk") as csvfile:
                reader = csv.DictReader(csvfile)
                for i, row in enumerate(reader, 1):
                    try:
                        point = Point(config.INFLUX_MEASUREMENT_NAME)
                        for tag_key in config.TAG_COLUMNS:
                            point.tag(tag_key, row.get(tag_key))

                        for field_key in config.FIELD_COLUMNS:
                            try:
                                point.field(field_key, float(row[field_key]))
                            except (ValueError, TypeError, KeyError):
                                continue  # 跳过空或格式错误的字段

                        naive_dt = datetime.datetime.strptime(
                            row[config.TIMESTAMP_COLUMN], config.TIMESTAMP_FORMAT
                        )
                        point.time(naive_dt.replace(tzinfo=local_tz))

                        points_buffer.append(point)
                        if len(points_buffer) >= batch_size:
                            write_api.write(bucket=config.INFLUXDB_BUCKET, record=points_buffer)
                            total_count += len(points_buffer)
                            print(f"  ...已写入 {total_count} 行...")
                            points_buffer.clear()

                    except Exception as e:
                        print(f"    处理文件 {csv_filepath} 的第 {i} 行时出错: {e}")

                if points_buffer:  # 写入最后一批
                    write_api.write(bucket=config.INFLUXDB_BUCKET, record=points_buffer)
                    total_count += len(points_buffer)

            print(
                f"✅ 文件 '{os.path.basename(csv_filepath)}' 导入成功！总共处理了 {total_count} 行数据。"
            )

        except FileNotFoundError:
            print(f"错误：找不到文件 '{csv_filepath}'。")
        except Exception as e:
            print(f"错误：读取或写入文件 '{csv_filepath}' 时发生严重错误: {e}")


def get_full_table_from_clickhouse(
    client: Client,
    database_name: str,
    table_name: str,
    start_time: datetime.datetime = None,
    end_time: datetime.datetime = None,
) -> pd.DataFrame:
    """
    从 ClickHouse 提取【指定时间范围】的【整张表】数据。
    如果时间范围为 None，则提取全部数据（但可能导致内存溢出）。
    """
    # print(f"  ► 正在从 ClickHouse '{database_name}'.`{table_name}` 提取数据...")

    full_table_path = f"`{database_name}`.`{table_name}`"

    where_conditions = []
    params = {}

    # 原始表的时间列
    time_col = "time"

    if start_time:
        where_conditions.append(f"`{time_col}` >= %(start)s")
        params["start"] = start_time.astimezone(datetime.timezone.utc)
    if end_time:
        where_conditions.append(f"`{time_col}` < %(end)s")
        params["end"] = end_time.astimezone(datetime.timezone.utc)

    where_clause = ""
    if where_conditions:
        where_clause = "WHERE " + " AND ".join(where_conditions)

    query = f"SELECT * FROM {full_table_path} {where_clause} ORDER BY `{time_col}`"

    try:
        df = client.query_dataframe(query, params=params)

        if df.empty:
            print(f"  ► 警告：在指定时间范围内，表 '{full_table_path}' 为空。")
            return pd.DataFrame()

        if time_col not in df.columns:
            print(f"❌ 错误：从ClickHouse返回的数据中找不到 '{time_col}' 列。")
            return pd.DataFrame()
        # 转换为 Pandas Datetime 对象
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        # 如果有重复时间戳，取平均值
        # df = df.groupby(time_col).mean(numeric_only=True)
        # 将其本地化为 UTC
        # df.index = df.index.tz_localize("UTC")

        return df

    except Exception as e:
        print(f"❌  从 {full_table_path} 提取数据时出错: {e}")
        raise


def get_data_from_clickhouse(
    client: Client,
    database_name: str,
    table_name: str,
    temple_id: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> pd.DataFrame:
    """
    按【时间范围】和【temple_id】从 ClickHouse 提取数据。
    """

    full_table_path = f"`{database_name}`.`{table_name}`"
    where_conditions = []
    params = {}
    time_col = "time"

    # 1. 时间范围筛选
    where_conditions.append(f"`{time_col}` >= %(start)s")
    params["start"] = start_time.astimezone(datetime.timezone.utc)
    where_conditions.append(f"`{time_col}` < %(end)s")
    params["end"] = end_time.astimezone(datetime.timezone.utc)

    # 2. 窟号 (Temple ID) 筛选
    where_conditions.append("`temple_id` = %(temple)s")  # 假设列名为 'temple_id'
    params["temple"] = temple_id

    where_clause = "WHERE " + " AND ".join(where_conditions)

    # 3. 查询
    # 我们需要 'time', 'device_id' 以及所有数值列 (如 humidity, temperature)
    query = f"SELECT * FROM {full_table_path} {where_clause} ORDER BY `device_id`, `{time_col}`"

    try:
        df = client.query_dataframe(query, params=params)

        if df.empty:
            return pd.DataFrame()

        # 确保 time 列是带时区的 datetime 对象
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        return df

    except Exception as e:
        print(f"❌ ClickHouse 查询失败: {e}")
        return pd.DataFrame()


def load_to_clickhouse(client: Client, df: pd.DataFrame, table_name: str) -> bool:
    """
    通用的 ClickHouse 数据加载函数，根据 DataFrame 的列名自动插入。
    """
    if df.empty:
        print(f"数据为空，跳过插入表 '{table_name}'")
        return False

    try:
        data_to_insert = df.to_dict("records")
        # 构造插入语句，使用反引号 ` ` 保证字段名正确
        column_names = ", ".join([f"`{col}`" for col in df.columns])
        query = f"INSERT INTO {table_name} ({column_names}) VALUES"
        client.execute(query, data_to_insert)
        print(f"成功: {len(data_to_insert)} 条记录已插入到表 '{table_name}'")
        return True
    except Exception as e:
        print(f"错误: 插入数据到 '{table_name}' 失败: {e}")
        if data_to_insert:
            print(f"失败数据示例 (第一行): {data_to_insert[0]}")
        return False


def store_dataframe_to_clickhouse(
    df: pd.DataFrame, client: Client, database_name: str, table_name: str, chunk_size: int = 100000
):
    """
    将 Pandas DataFrame 【分块】写入【指定的数据库】和【指定的表】。
    """
    if df.empty:
        print("\n  ► 数据为空，跳过写入 ClickHouse。")
        return

    full_table_path = f"`{database_name}`.`{table_name}`"

    try:
        df_to_insert = df.copy()
        order_by_col_name = ""  # 需要变量来存储【清理后】的排序列名

        if isinstance(df_to_insert.index, pd.DatetimeIndex):
            index_name = df_to_insert.index.name if df_to_insert.index.name else "timestamp"
            df_to_insert = df_to_insert.reset_index().rename(columns={index_name: index_name})
            order_by_col_name = index_name  # 原始排序列名 (例如 '_time')
        else:
            order_by_col_name = df_to_insert.columns[0]

        #  ===== 先清理所有列名 =====
        sanitized_columns = [re.sub(r"\W+", "_", col).strip("_") for col in df_to_insert.columns]
        df_to_insert.columns = sanitized_columns

        # 我们也清理原始的排序列名，确保它和新列名能对上
        sanitized_order_by_col = re.sub(r"\W+", "_", order_by_col_name).strip("_")
        order_by_clause = f"(`{sanitized_order_by_col}`)"

        # 自动生成建表 SQL 模板
        # 此时 df_to_insert (列名已清理) 和 order_by_clause (也已清理)
        # 传递给 generate_create_table_sql 时是完全匹配的
        schema_template = generate_create_table_sql(
            df_to_insert,
            full_table_path,
            engine="MergeTree",
            order_by=order_by_clause,  # 传入清理后的 "(`time`)"
        )

        # 调用建表函数
        if not create_table_if_not_exists(client, database_name, table_name, schema_template):
            raise Exception("建表失败，请检查错误信息。")

        # 分块插入数据
        column_names_str = ", ".join([f"`{col}`" for col in df_to_insert.columns])
        query = f"INSERT INTO {full_table_path} ({column_names_str}) VALUES"

        num_chunks = int(np.ceil(len(df_to_insert) / chunk_size))
        for i, chunk_df in enumerate(np.array_split(df_to_insert, num_chunks)):
            data_to_insert = chunk_df.values.tolist()
            client.execute(query, data_to_insert)

    except Exception as e:
        print(f"❌  写入 ClickHouse 时发生错误: {e}")
        raise


# ====================================================================
#   高级工作流
# ====================================================================


def process_excel_file(file_path: str, client: Client, rules_config: dict, parse_config: dict):
    """
    处理单个传感器数据 Excel 文件：解析、转换并存入 ClickHouse。
    这是一个高级工作流，组合了文件读取、数据转换和数据库写入。
    """
    filename = os.path.basename(file_path)
    match = re.match(parse_config["filename_regex"], filename)
    if not match:
        print(f"  文件名 {filename} 不符合规则，跳过。")
        return

    # 从文件名提取元数据
    device_id, temple_id, keyword = match.groups()

    if keyword not in rules_config:
        print(f"  未找到关键字 '{keyword}' 的映射规则，跳过。")
        return

    rules = rules_config[keyword]
    target_table = rules["clickhouse_table"]
    column_map = rules["column_mapping"]

    print(f"  解析成功: [Device: {device_id}, Temple: {temple_id}] -> 存入表: {target_table}")

    # 读取 Excel (处理多年份 Sheet)
    try:
        xls = pd.ExcelFile(file_path)
        header_row = parse_config["excel_reading"]["header_row"]
        start_year, end_year = parse_config["excel_reading"]["sheet_year_range"]
        valid_sheets = [
            s for s in xls.sheet_names if s.isdigit() and start_year <= int(s) <= end_year
        ]

        if not valid_sheets:
            print("  未找到任何有效的年份 Sheet，跳过。")
            return

        df_total = pd.concat(
            [pd.read_excel(xls, sheet_name=s, header=header_row) for s in valid_sheets],
            ignore_index=True,
        )

    except Exception as e:
        print(f"  读取或合并 Excel 文件失败 (可能文件已打开或损坏): {e}")
        return

    # 数据清洗和转换
    df_total.rename(columns=column_map, inplace=True)
    final_db_columns = list(column_map.values())
    df_to_insert = df_total[final_db_columns].copy()

    df_to_insert["device_id"] = device_id
    df_to_insert["temple_id"] = temple_id
    df_to_insert["created_at"] = datetime.datetime.now()
    df_to_insert["time"] = pd.to_datetime(df_to_insert["time"], errors="coerce")

    for col in final_db_columns:
        if col != "time":
            df_to_insert[col] = pd.to_numeric(df_to_insert[col], errors="coerce")

    df_to_insert.dropna(inplace=True)
    if df_to_insert.empty:
        print("  处理后数据为空 (可能所有行都有无效的时间或值)，跳过。")
        return

    # 准备数据以插入 ClickHouse
    try:
        full_table_name = f"{config.CLICKHOUSE_SHARED_DB}.{target_table}"
        co2_table_name = config.RAW_SENSOR_MAPPING_CONFIG["无线二氧化碳传感器"]["clickhouse_table"]

        # 确保列顺序与数据库表一致
        if target_table == co2_table_name:
            final_columns_order = [
                "temple_id",
                "device_id",
                "time",
                "co2_collected",
                "co2_corrected",
                "created_at",
            ]
        else:
            final_columns_order = [
                "temple_id",
                "device_id",
                "time",
                "humidity",
                "temperature",
                "created_at",
            ]

        df_to_insert = df_to_insert[final_columns_order]
        data_to_insert = df_to_insert.values.tolist()

        client.execute(f"INSERT INTO {full_table_name} VALUES", data_to_insert)
        print(f"  成功! 插入 {len(data_to_insert)} 行数据到 {full_table_name}。")

    except Exception as e:
        print(f"  插入数据到 ClickHouse 失败: {e}")
