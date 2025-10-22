# 与数据库连接、读取和写入相关的操作,包括从influxdb读取数据,以及将计算好的特征存入clickhouse

import csv
import datetime
import os
import re
import uuid
from zoneinfo import ZoneInfo

from clickhouse_driver import Client
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import numpy as np
import pandas as pd

from src import config
from src.utils import timing_decorator


## influxdb
def import_csv_to_influx(csv_filepath: str):
    """
    读取指定的 CSV 文件，并将其内容导入到 InfluxDB。
    """
    # 将配置项从 config 模块中读入
    local_tz = ZoneInfo(config.LOCAL_TIMEZONE)

    # 建立 InfluxDB 连接
    with InfluxDBClient(
        url=config.INFLUXDB_URL, token=config.INFLUXDB_TOKEN, org=config.INFLUXDB_ORG
    ) as client:
        write_api = client.write_api(write_options=SYNCHRONOUS)
        points_buffer = []
        batch_size = 5000  # 批处理大小
        count = 0
        total_count = 0

        try:
            with open(csv_filepath, "r", encoding="gbk") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        point = Point(config.INFLUX_MEASUREMENT_NAME)

                        for tag_key in config.TAG_COLUMNS:
                            point.tag(tag_key, row.get(tag_key))

                        for field_key in config.FIELD_COLUMNS:
                            try:
                                field_value = float(row[field_key])
                                point.field(field_key, field_value)
                            except (ValueError, TypeError, KeyError):
                                continue  # 如果某个字段为空或格式错误，跳过该字段

                        naive_dt = datetime.strptime(
                            row[config.TIMESTAMP_COLUMN], config.TIMESTAMP_FORMAT
                        )
                        aware_dt = naive_dt.replace(tzinfo=local_tz)
                        point.time(aware_dt)

                        points_buffer.append(point)
                        count += 1
                        total_count += 1

                        if count >= batch_size:
                            write_api.write(
                                bucket=config.INFLUXDB_BUCKET,
                                org=config.INFLUXDB_ORG,
                                record=points_buffer,
                            )
                            print(f"  ...已写入 {total_count} 行...")
                            points_buffer = []
                            count = 0

                    except Exception as e:
                        print(f"    处理文件 {csv_filepath} 的第 {total_count + 1} 行时出错: {e}")

                # 写入最后一批不足 batch_size 的数据
                if points_buffer:
                    write_api.write(
                        bucket=config.INFLUXDB_BUCKET,
                        org=config.INFLUXDB_ORG,
                        record=points_buffer,
                    )

                print(
                    f"✅ 文件 '{os.path.basename(csv_filepath)}' 导入成功！总共处理了 {total_count} 行数据。"
                )

        except FileNotFoundError:
            print(f"错误：找不到文件 '{csv_filepath}'。")
        except Exception as e:
            print(f"错误：读取或写入文件 '{csv_filepath}' 时发生严重错误: {e}")


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


## clickhouse


def get_clickhouse_client(database_name):
    """
    尝试连接到指定的 ClickHouse 数据库。
    """
    try:
        client = Client(
            host=config.CLICKHOUSE_SHARED_HOST,
            port=config.CLICKHOUSE_SHARED_PORT,
            user=config.CLICKHOUSE_SHARED_USER,
            password=config.CLICKHOUSE_SHARED_PASSWORD,
            database=database_name,  # <-- 使用参数
        )
        print(
            f"成功连接到 ClickHouse {config.CLICKHOUSE_SHARED_HOST}:{config.CLICKHOUSE_SHARED_PORT} (数据库: {database_name})"
        )
        return client
    except Exception as e:
        # 打印错误，但让调用者（主脚本）来处理异常
        print(f"连接 ClickHouse 数据库 '{database_name}' 失败: {e}")
        raise  # 重新抛出异常


# def store_features_to_clickhouse(
#     df: pd.DataFrame,
#     table_name: str,
#     field_name: str,
#     device_id: str,
#     temple_id: str,
#     stats_cycle: str,
# ):
#     """
#     接收【宽格式】的特征 DataFrame，为其生成唯一ID，转换为【长格式】，并存入 ClickHouse。
#     """
#     if df.empty:
#         print("\n特征数据为空，跳过存储。")
#         return

#     print(f"\n开始处理并存入 ClickHouse 的表: '{table_name}'...")

#     try:
#         df_long = df.reset_index()
#         feature_columns = [col for col in df.columns if col not in ["分析周期"]]
#         df_long = pd.melt(
#             df_long,
#             id_vars=["_time"],
#             value_vars=feature_columns,
#             var_name="feature_key",
#             value_name="feature_value",
#         )

#         df_long["stat_id"] = [uuid.uuid4() for _ in range(len(df_long))]
#         df_long["temple_id"] = temple_id
#         df_long["device_id"] = device_id
#         df_long["monitored_variable"] = field_name
#         df_long["stats_cycle"] = stats_cycle
#         df_long["created_at"] = datetime.datetime.now()
#         df_long.rename(columns={"_time": "stats_start_time"}, inplace=True)

#         final_columns = [
#             "stat_id",
#             "temple_id",
#             "device_id",
#             "stats_start_time",
#             "monitored_variable",
#             "stats_cycle",
#             "feature_key",
#             "feature_value",
#             "created_at",
#         ]
#         df_to_insert = df_long[final_columns]
#         df_to_insert["feature_value"] = pd.to_numeric(
#             df_to_insert["feature_value"], errors="coerce"
#         )
#         df_to_insert["standby_field01"] = ""

#         client = clickhouse_connect.get_client(
#             host=config.CLICKHOUSE_HOST,
#             port=config.CLICKHOUSE_PORT,
#             username=config.CLICKHOUSE_USER,
#             password=config.CLICKHOUSE_PASSWORD,
#         )
#         client.command(f"CREATE DATABASE IF NOT EXISTS {config.DATABASE_NAME}")
#         create_table_query = f"""
#         CREATE TABLE IF NOT EXISTS {config.DATABASE_NAME}.`{table_name}`
#         (
#             `stat_id` UUID,
#             `temple_id` CHAR(20),
#             `device_id` CHAR(30),
#             `stats_start_time` DATETIME,
#             `monitored_variable` CHAR(20),
#             `stats_cycle` CHAR(10),
#             `feature_key` CHAR(30),
#             `feature_value` VARCHAR(30),
#             `standby_field01` CHAR(30),
#             `created_at` TIMESTAMP
#         ) ENGINE = MergeTree() ORDER BY (stats_start_time, device_id, feature_key)
#         """
#         client.command(create_table_query)

#         # 在存入 ClickHouse 前，先将数据保存到 CSV 文件
#         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#         csv_filename = f"{table_name}_{timestamp}.csv"
#         print(f"  ► 正在将数据保存到 CSV 文件: {csv_filename} ...")
#         try:
#             df_to_insert.to_csv(
#                 csv_filename, index=False, encoding="utf-8-sig", float_format="%.2f"
#             )
#             print(f"  ✔ CSV 文件保存成功: {csv_filename}")
#         except Exception as csv_error:
#             print(f"  ❌ 保存 CSV 文件时出错: {csv_error}")

#         # 写入CSV之后，将数值转换为格式化的字符串
#         df_to_insert["feature_value"] = df_to_insert["feature_value"].map("{:.2f}".format)
#         # 处理可能因coerce产生的NaN值，将其转换为空字符串
#         df_to_insert["feature_value"] = df_to_insert["feature_value"].replace("nan", "")

#         print(f"  ...正在插入 {len(df_to_insert)} 行长格式特征数据...")
#         client.insert_df(f"{config.DATABASE_NAME}.`{table_name}`", df_to_insert)

#     except Exception as e:
#         print(f"❌ 存入 ClickHouse 时发生错误: {e}")


def _create_raw_tables_if_not_exists(client):
    """
    创建原始传感器数据表（如果不存在）。
    """
    db_name = config.CLICKHOUSE_SHARED_DB

    # 1. 温湿度表 (从 config 获取表名)
    table_temp = config.RAW_SENSOR_MAPPING_CONFIG["无线温湿度传感器"]["clickhouse_table"]
    create_temp_query = f"""
    CREATE TABLE IF NOT EXISTS {db_name}.{table_temp} (
        temple_id     CHAR(20),
        device_id     CHAR(30),
        time          DATETIME,
        humidity      FLOAT(32),
        temperature   FLOAT(32),
        created_at    TIMESTAMP
    ) ENGINE = MergeTree()
    PARTITION BY toYYYYMM(time)
    ORDER BY (temple_id, device_id, time)
    """

    # 2. CO2 表 (从 config 获取表名)
    table_co2 = config.RAW_SENSOR_MAPPING_CONFIG["无线二氧化碳传感器"]["clickhouse_table"]
    create_co2_query = f"""
    CREATE TABLE IF NOT EXISTS {db_name}.{table_co2} (
        temple_id     CHAR(20),
        device_id     CHAR(30),
        time          DATETIME,
        co2_collected FLOAT(32),
        co2_corrected FLOAT(32),
        created_at    TIMESTAMP
    ) ENGINE = MergeTree()
    PARTITION BY toYYYYMM(time)
    ORDER BY (temple_id, device_id, time)
    """

    try:
        client.execute(create_temp_query)
        print(f"表 {db_name}.{table_temp} 检查/创建 成功。")
        client.execute(create_co2_query)
        print(f"表 {db_name}.{table_co2} 检查/创建 成功。")
    except Exception as e:
        print(f"创建表失败: {e}")
        raise


def process_excel_file(file_path, client, rules_config, parse_config):
    """
    处理单个 Excel 文件：解析、读取所有年份 Sheet、并插入数据库。
    """
    filename = os.path.basename(file_path)

    # 解析文件名 (使用 config 中的 RAW_FILENAME_REGEX)
    match = re.match(parse_config["filename_regex"], filename)
    if not match:
        print(f"  文件名 {filename} 不符合规则，跳过。")
        return

    # 提取信息
    device_id = match.group(1)  # 按您要求，保持原始大小写
    temple_id = match.group(2)
    keyword = match.group(3)

    # 获取映射规则 (使用 config 中的 RAW_SENSOR_MAPPING_CONFIG)
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
    except Exception as e:
        print(f"  读取 Excel 文件失败 (可能文件已打开或损坏): {e}")
        return

    all_sheets_data = []
    excel_options = parse_config["excel_reading"]
    header_row = excel_options["header_row"]
    start_year, end_year = excel_options["sheet_year_range"]

    # 筛选出在年份范围内的所有 Sheet
    valid_sheet_names = [
        s for s in xls.sheet_names if s.isdigit() and start_year <= int(s) <= end_year
    ]

    if not valid_sheet_names:
        print("  未找到任何有效的年份，跳过。")
        return

    for sheet_name in valid_sheet_names:
        try:
            df_sheet = pd.read_excel(xls, sheet_name=sheet_name, header=header_row)
            all_sheets_data.append(df_sheet)
        except Exception as e:
            print(f"    读取 Sheet '{sheet_name}' 失败: {e}")

    if not all_sheets_data:
        print("  所有 Sheet 读取失败，未获取任何数据。")
        return

    # 合并、处理数据
    df_total = pd.concat(all_sheets_data, ignore_index=True)
    df_total.rename(columns=column_map, inplace=True)
    final_db_columns = list(column_map.values())
    df_to_insert = df_total[final_db_columns].copy()

    df_to_insert["device_id"] = device_id
    df_to_insert["temple_id"] = temple_id
    df_to_insert["created_at"] = datetime.datetime.now()

    df_to_insert["time"] = pd.to_datetime(df_to_insert["time"], errors="coerce")
    df_to_insert.dropna(subset=["time"], inplace=True)

    # 将数值列转为
    for col in final_db_columns:
        if col != "time":
            df_to_insert[col] = pd.to_numeric(df_to_insert[col], errors="coerce")

    # 删除任何包含空值(NaN)的行，因为 ClickHouse FLOAT 不接受 NaN
    df_to_insert.dropna(inplace=True)
    if df_to_insert.empty:
        print("  处理后数据为空 (可能所有行都有无效的时间或值)，跳过。")
        return

    try:
        full_table_name = f"{config.CLICKHOUSE_SHARED_DB}.{target_table}"
        # 转换数据为列表 (clickhouse-driver 推荐方式)
        # 必须确保 DataFrame 的列顺序与 ClickHouse 表一致
        co2_table_name = config.RAW_SENSOR_MAPPING_CONFIG["无线二氧化碳传感器"]["clickhouse_table"]
        if target_table == co2_table_name:
            final_columns_order = [
                "temple_id",
                "device_id",
                "time",
                "co2_collected",
                "co2_corrected",
                "created_at",
            ]
        else:  # RAW_TABLE_TEMP_HUMIDITY
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

        insert_query = f"INSERT INTO {full_table_name} VALUES"

        # client.execute 会自动处理批量插入
        client.execute(insert_query, data_to_insert)

        print(f"  成功! 插入 {len(data_to_insert)} 行数据到 {full_table_name}。")

    except Exception as e:
        print(f"  插入数据到 ClickHouse 失败: {e}")


def _create_table_if_not_exists(client: Client, db_name: str, table_name: str):
    """
    创建特征数据表（如果不存在）。
    """

    client.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {db_name}.`{table_name}`
    (
        `stat_id`            BIGINT,
        `temple_id`          FixedString(20),
        `device_id`          FixedString(30),
        `stats_start_time`   DateTime,
        `monitored_variable` FixedString(20),
        `stats_cycle`        FixedString(10),
        `feature_key`        FixedString(30),
        `feature_value`      FixedString(30),  
        `standby_field01`    FixedString(30),
        `created_at`         DateTime          
    ) ENGINE = MergeTree()
    ORDER BY (stats_start_time, device_id, feature_key)
    """
    client.execute(create_table_query)
    print(f"表 {db_name}.`{table_name}` 检查/创建完毕 。")


def store_dataframe_to_clickhouse(
    df: pd.DataFrame,
    table_name: str,
    target: str = "shared",
    chunk_size: int = 500000,
):
    """
    将 DataFrame 存储到 ClickHouse (使用 clickhouse-driver TCP 协议)。
    """
    if df.empty:
        print("\n数据为空，跳过存储。")
        return

    if target == "shared":
        host = config.CLICKHOUSE_SHARED_HOST
        port = config.CLICKHOUSE_SHARED_PORT
        user = config.CLICKHOUSE_SHARED_USER
        password = config.CLICKHOUSE_SHARED_PASSWORD
        database = config.CLICKHOUSE_SHARED_DB
    else:
        raise ValueError(f"未知的 ClickHouse 目标: {target}")

    full_table_name = f"{database}.{table_name}"
    print(f"准备连接到 {host}:{port}，存入 {full_table_name}...")

    df_to_insert = df.copy()
    df_to_insert.reset_index(inplace=True)
    if config.FIELD_NAME not in df_to_insert.columns:
        raise KeyError(f"DataFrame 中找不到值列: '{config.FIELD_NAME}'")
    if "_time" not in df_to_insert.columns:
        raise KeyError("DataFrame 中找不到时间列: '_time' (请检查 reset_index)")

    df_to_insert.rename(
        columns={"_time": "stats_start_time", config.FIELD_NAME: "feature_value"},
        inplace=True,
    )
    # 填充列
    start_int_id = int(datetime.datetime.now().timestamp() * 1_000_000)
    df_to_insert["stat_id"] = range(start_int_id, start_int_id + len(df_to_insert))
    df_to_insert["temple_id"] = config.TEMPLE_ID
    df_to_insert["device_id"] = config.DEVICE_ID
    df_to_insert["created_at"] = datetime.datetime.now()

    final_columns = [
        "stat_id",
        "temple_id",
        "device_id",
        "stats_start_time",
        "monitored_variable",
        "stats_cycle",
        "feature_key",
        "feature_value",
        "standby_field01",
        "created_at",
    ]

    # 检查是否有缺失的列
    missing_cols = set(final_columns) - set(df_to_insert.columns)
    if missing_cols:
        print(f"警告: DataFrame 中缺少以下列: {missing_cols}")
        # 补全缺失的列为空值，防止插入失败
        for col in missing_cols:
            df_to_insert[col] = ""  # 假设 CHAR 类型可以接受空字符串

    df_to_insert = df_to_insert[final_columns]

    try:
        client = Client(host=host, port=port, user=user, password=password, database="default")
        print("连接成功。")
        _create_table_if_not_exists(client, database, table_name)
        # 准备 INSERT 语句
        insert_query = f"INSERT INTO {full_table_name} VALUES"
        # 转换数据 (使用 .values.tolist() 兼容性好)
        data_to_insert = df_to_insert.values.tolist()
        total_rows = len(data_to_insert)
        chunk_size = int(chunk_size)
        for i in range(0, total_rows, chunk_size):
            chunk = data_to_insert[i : i + chunk_size]
            client.execute(insert_query, chunk)

        print(f"成功插入 {total_rows} 行数据到 {full_table_name}。")

    except Exception as e:
        print(f"存储到 ClickHouse 失败: {e}")
        raise e


## 预处理
def preprocess_timeseries_data(
    df: pd.DataFrame,
    resample_freq: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> pd.DataFrame:
    """
    对原始时间序列数据进行标准化预处理：
    1. 确保时间索引是UTC时区。
    2. 按照指定频率和起止时间，创建完整的、连续的时间轴。
    3. 优先使用线性插值 (interpolate) 填充内部缺失值，
       然后使用 ffill 和 bfill 填充边缘（开头和结尾）的缺失值。
    4. 将最终的时间索引转换为本地时区（东八区）。
    """
    if df.empty:
        return df

    if df.index.tz is None:
        df = df.tz_localize("UTC")
    else:
        df = df.tz_convert("UTC")

    # 确保 start_time 和 end_time 也是 UTC
    start_utc = start_time.astimezone(datetime.timezone.utc)
    end_utc = end_time.astimezone(datetime.timezone.utc)

    full_range = pd.date_range(start=start_utc, end=end_utc, freq=resample_freq)
    reindexed_df = df.reindex(full_range)

    #    使用线性插值填充所有 "内部" 的 NaNs
    interpolated_df = reindexed_df.interpolate(method="linear")

    #    使用 ffill 和 bfill 填充可能残留在
    filled_df = interpolated_df.ffill().bfill()

    final_df = filled_df.tz_convert(config.LOCAL_TIMEZONE)
    final_df.index.name = "_time"

    print(f"  ► 处理完成。处理后的数据共有 {len(final_df)} 条记录。")
    return final_df


def preprocess_limited_interpolation(
    df: pd.DataFrame,
    resample_freq: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    gap_threshold_hours: float = 2.0,
) -> pd.DataFrame:
    """
    对时间序列数据进行预处理，并仅对短于门限的空缺进行线性插值。

    1. 确保时间索引是UTC时区。
    2. 按照指定频率和起止时间，创建完整的、连续的时间轴。
    3. 仅对 (gap_threshold_hours) 或更短的连续缺失值执行线性插值。
    4. 较长的缺失值将保留为 NaNs。
    """
    if df.empty:
        return df

    if df.index.tz is None:
        df = df.tz_localize("UTC")
    else:
        df = df.tz_convert("UTC")

    start_utc = start_time.astimezone(datetime.timezone.utc)
    end_utc = end_time.astimezone(datetime.timezone.utc)

    full_range = pd.date_range(start=start_utc, end=end_utc, freq=resample_freq)
    reindexed_df = df.reindex(full_range)

    try:
        threshold_duration = pd.Timedelta(hours=gap_threshold_hours)
        resample_duration = pd.Timedelta(resample_freq)
    except ValueError:
        raise ValueError(
            f"无效的 resample_freq: '{resample_freq}'。请使用 '1T', '5min', '1H' 等。"
        )

    if resample_duration.total_seconds() <= 0:
        raise ValueError(f"resample_freq '{resample_freq}' 必须是正的时间间隔。")

    limit_count = int(threshold_duration / resample_duration) - 1

    if limit_count < 1:
        print(
            f"警告: resample_freq ({resample_freq}) 大于或等于门限 ({gap_threshold_hours}h)。将不进行插值。"
        )
        limit_count = 0

    interpolated_df = reindexed_df.interpolate(method="linear", limit=limit_count)
    final_df = interpolated_df.tz_convert(config.LOCAL_TIMEZONE)
    final_df.index.name = "_time"

    print(f"  ► 预处理(有限插值)完成。处理后的数据共有 {len(final_df)} 条记录。")
    return final_df
