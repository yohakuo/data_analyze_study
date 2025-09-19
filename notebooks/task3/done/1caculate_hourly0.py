import datetime
import time
from zoneinfo import ZoneInfo

import clickhouse_connect
import pandas as pd
from influxdb_client import InfluxDBClient

# --- 1. InfluxDB 连接配置 ---
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "study2025"
INFLUXDB_ORG = "task3"
INFLUXDB_BUCKET = "cave45"

# # --- 2. 查询参数配置  ---
# LOCAL_TIMEZONE = "Asia/Shanghai"
# START_YEAR = 2021
# START_MONTH = 1
# START_DAY = 1

# STOP_YEAR = 2021
# STOP_MONTH = 1
# STOP_DAY = 2 # 查询截止到 1 月 2 日的 0 点，正好是一整天的数据

MEASUREMENT_NAME = "adata"
FIELD_NAME = "空气湿度"

"""
从 InfluxDB 查询全部湿度数据，并返回一个清理好的 Pandas DataFrame.
"""


def get_humidity_data() -> pd.DataFrame:
    with InfluxDBClient(
        url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG
    ) as client:
        # 使用 range(start: 0) 获取所有数据
        query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: 0)
          |> filter(fn: (r) => r["_measurement"] == "{MEASUREMENT_NAME}")
          |> filter(fn: (r) => r["_field"] == "{FIELD_NAME}")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''

        print("正在从 InfluxDB 查询【所有】数据，请稍候...")
        df = client.query_api().query_data_frame(query=query, org=INFLUXDB_ORG)

        if df.empty:
            print("警告：查询结果为空，请检查 InfluxDB 中是否有数据。")
            return df

        print("数据查询成功，正在进行预处理...")

        if FIELD_NAME in df.columns and "_time" in df.columns:
            df_cleaned = df[["_time", FIELD_NAME]]
            df_cleaned = df_cleaned.set_index("_time")
            df_cleaned.index = pd.to_datetime(df_cleaned.index)
            df_cleaned[FIELD_NAME] = pd.to_numeric(
                df_cleaned[FIELD_NAME], errors="coerce"
            )
            df_cleaned = df_cleaned.dropna()
            return df_cleaned
        else:
            print("错误：返回的数据中缺少 '_time' 或 '空气湿度' 列。")


"""
接收一个小时的原始数据序列(series)，计算超过Q3的数据点占比。
"""


def calculate_percent_above_q3(series):
    if len(series) < 4:
        return 0.0
    q3 = series.quantile(0.75)
    # 避免 Q3 等于最大值时，没有数能大于 Q3 的情况
    if q3 == series.max():
        return 0.0
    count_above_q3 = (series > q3).sum()
    percent_above_q3 = count_above_q3 / len(series)
    return percent_above_q3


"""
接收原始数据 DataFrame，计算小时级统计特征，并返回结果 DataFrame.
"""


def calculate_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    print("\n开始计算小时级统计特征...")

    # 1. 先计算基础特征
    hourly_stats = df.resample("h").agg(
        均值=("空气湿度", "mean"),
        中位数=("空气湿度", "median"),
        最大值=("空气湿度", "max"),
        最小值=("空气湿度", "min"),
        Q1=("空气湿度", lambda x: x.quantile(0.25)),
        Q3=("空气湿度", lambda x: x.quantile(0.75)),
        P10=("空气湿度", lambda x: x.quantile(0.10)),
    )
    hourly_stats["极差"] = hourly_stats["最大值"] - hourly_stats["最小值"]

    # 2. 计算高级特征一：“超过 Q3 的占时比”
    # 我们对按小时分组的数据，应用(apply)我们的自定义函数
    percent_above_q3 = df["空气湿度"].resample("h").apply(calculate_percent_above_q3)
    # 将计算结果合并到我们的特征表里
    hourly_stats["超过Q3占时比"] = percent_above_q3

    # 3. 计算高级特征二：“极差的时间变化率”
    # 使用 .pct_change() 计算与上一行的变化率，并用 0 填充第一个无法计算的 NaN 值
    hourly_stats["极差的时间变化率"] = hourly_stats["极差"].pct_change().fillna(0)

    # 4. 最后整理一下列名
    hourly_stats.rename(
        columns={"中位数": "中位数 (Q2)", "P10": "10th百分位数"}, inplace=True
    )

    print("所有小时级特征计算完成！")
    return hourly_stats


"""
将计算好的特征 DataFrame 存储到 ClickHouse 数据库中。
"""


def store_features_to_clickhouse(df: pd.DataFrame):
    if df.empty:
        print("\n特征数据为空，跳过存储。")
        return

    print("\n开始将特征数据存入 ClickHouse...")

    # --- ClickHouse 连接配置 ---
    CLICKHOUSE_HOST = "localhost"
    CLICKHOUSE_PORT = 8123
    DATABASE_NAME = "feature_db"
    TABLE_NAME = "humidity_hourly_features"

    try:
        # 1. 连接到 ClickHouse
        # client.command() 用于执行非查询类的 SQL 语句
        client = clickhouse_connect.get_client(
            host=CLICKHOUSE_HOST,
            port=CLICKHOUSE_PORT,
            username="default",
            password="study2025",
        )
        # 2. 创建数据库 (如果不存在的话)
        client.command(f"CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}")

        # 3. 定义建表语句
        # 我们需要根据 DataFrame 的列来精心设计表的结构
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {DATABASE_NAME}.{TABLE_NAME}
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
        print(f"数据库 '{DATABASE_NAME}' 和表 '{TABLE_NAME}' 已准备就绪。")
        # 4. 准备数据用于插入
        # a. 复制一份数据，避免修改原始的 DataFrame
        df_to_insert = df.copy()

        # b. 添加你最初要求的描述性字段
        df_to_insert["分析周期"] = "hourly"

        # c. 把索引（时间）变回普通列，并重命名以匹配表结构
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

        # 5. 插入数据
        print(f"正在插入 {len(df_to_insert)} 行数据...")
        client.insert_df(f"{DATABASE_NAME}.{TABLE_NAME}", df_to_insert)

        print("数据成功存入 ClickHouse！")

    except Exception as e:
        print(f"存入 ClickHouse 时发生错误: {e}")


if __name__ == "__main__":
    # humidity_df = get_humidity_data()

    # if not humidity_df.empty:
    #     print("\n数据准备完成！预览前5条数据：")
    #     print(humidity_df.head())

    #     print(f"\n成功提取了 {len(humidity_df)} 条湿度数据。")
    #     print(f"数据时间范围从 {humidity_df.index.min()} 到 {humidity_df.index.max()}")

    #     hourly_features = calculate_hourly_features(humidity_df)

    #     if not hourly_features.empty:
    #         print("\n===== 小时级特征计算结果预览：=====")
    #         # .head(24) 可以显示第一天的24个小时的结果
    #         print(hourly_features.head(24))

    #         store_features_to_clickhouse(hourly_features)
    pipeline_start_time = time.perf_counter()
    print("===== 数据提取与准备 =====")
    extraction_start_time = time.perf_counter()

    humidity_df = get_humidity_data()

    extraction_end_time = time.perf_counter()
    extraction_duration = extraction_end_time - extraction_start_time
    print(f"✅ 从 InfluxDB 提取数据耗时: {extraction_duration:.2f} 秒")

    if not humidity_df.empty:
        print(f"    (共提取了 {len(humidity_df)} 条原始数据)")
        print("\n===== 特征工程计算 =====")
        features_start_time = time.perf_counter()

        hourly_features = calculate_hourly_features(humidity_df)

        features_end_time = time.perf_counter()
        features_duration = features_end_time - features_start_time
        print(f"计算小时级特征耗时: {features_duration:.2f} 秒")

        if not hourly_features.empty:
            print(f"    (共生成了 {len(hourly_features)} 条小时级特征)")
            # print(hourly_features.head(3)) # 如果不想看预览可以注释掉这行

            print("\n===== 数据存储 =====")
            storage_start_time = time.perf_counter()

            store_features_to_clickhouse(hourly_features)

            storage_end_time = time.perf_counter()
            storage_duration = storage_end_time - storage_start_time
            print(f"✅ 存入 ClickHouse 耗时: {storage_duration:.2f} 秒")

    # --- 总结 ---
    pipeline_end_time = time.perf_counter()
    pipeline_duration = pipeline_end_time - pipeline_start_time
    print(
        f"\n🎉 ** 数据处理流水线全部执行完毕！总耗时: {pipeline_duration:.2f} 秒 ** 🎉"
    )
