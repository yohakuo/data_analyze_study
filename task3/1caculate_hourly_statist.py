import pandas as pd
from influxdb_client import InfluxDBClient
import datetime
from zoneinfo import ZoneInfo
import clickhouse_connect
import time

# ---  InfluxDB 连接配置 ---
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "study2025"
INFLUXDB_ORG = "task3"
INFLUXDB_BUCKET = "cave45"
MEASUREMENT_NAME = 'adata'
FIELD_NAME = '空气湿度'

CLICKHOUSE_HOST = 'localhost'
CLICKHOUSE_PORT = 8123
CLICKHOUSE_USER = 'default'
CLICKHOUSE_PASSWORD = 'study2025'
DATABASE_NAME = 'feature_db'
TABLE_NAME = 'humidity_hourly_features'


def get_humidity_data(start_time_utc_str: str, stop_time_utc_str: str) -> pd.DataFrame:
    """
    从 InfluxDB 查询【指定时间范围】的湿度数据。
    """
    with InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG) as client:
        query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: {start_time_utc_str}, stop: {stop_time_utc_str})
          |> filter(fn: (r) => r["_measurement"] == "{MEASUREMENT_NAME}")
          |> filter(fn: (r) => r["_field"] == "{FIELD_NAME}")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        
        print(f"  正在从 InfluxDB 查询从 {start_time_utc_str} 到 {stop_time_utc_str} 的数据...")
        df = client.query_api().query_data_frame(query=query, org=INFLUXDB_ORG)
        
        if df.empty:
            return df
        
        # (数据清理部分不变)
        if FIELD_NAME in df.columns and '_time' in df.columns:
            df_cleaned = df[['_time', FIELD_NAME]]
            df_cleaned = df_cleaned.set_index('_time')
            df_cleaned.index = pd.to_datetime(df_cleaned.index)
            df_cleaned[FIELD_NAME] = pd.to_numeric(df_cleaned[FIELD_NAME], errors='coerce')
            df_cleaned = df_cleaned.dropna()
            return df_cleaned
        return pd.DataFrame()
        
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

def calculate_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    print("\n开始计算小时级统计特征...")
    
    # 1. 先计算基础特征
    hourly_stats = df.resample('h').agg(
        均值=('空气湿度', 'mean'),
        中位数=('空气湿度', 'median'),
        最大值=('空气湿度', 'max'),
        最小值=('空气湿度', 'min'),
        Q1=('空气湿度', lambda x: x.quantile(0.25)),
        Q3=('空气湿度', lambda x: x.quantile(0.75)),
        P10=('空气湿度', lambda x: x.quantile(0.10))
    )
    hourly_stats['极差'] = hourly_stats['最大值'] - hourly_stats['最小值']
    
    # 2. 计算高级特征一：“超过 Q3 的占时比”
    # 我们对按小时分组的数据，应用(apply)我们的自定义函数
    percent_above_q3 = df['空气湿度'].resample('h').apply(calculate_percent_above_q3)
    # 将计算结果合并到我们的特征表里
    hourly_stats['超过Q3占时比'] = percent_above_q3

    # 3. 计算高级特征二：“极差的时间变化率”
    # 使用 .pct_change() 计算与上一行的变化率，并用 0 填充第一个无法计算的 NaN 值
    hourly_stats['极差的时间变化率'] = hourly_stats['极差'].pct_change().fillna(0)

    # 4. 最后整理一下列名
    hourly_stats.rename(columns={'中位数': '中位数 (Q2)', 'P10': '10th百分位数'}, inplace=True)
    
    print("所有小时级特征计算完成！")
    return hourly_stats


def store_features_to_clickhouse(df: pd.DataFrame):
    if df.empty:
        print("\n特征数据为空，跳过存储。")
        return

    print("\n开始将特征数据存入 ClickHouse...")

    try:
        # 1. 连接到 ClickHouse
        # client.command() 用于执行非查询类的 SQL 语句
        client = clickhouse_connect.get_client(
            host=CLICKHOUSE_HOST, 
            port=CLICKHOUSE_PORT, 
            username='default', 
            password='study2025'  
        )        
        # 2. 创建数据库 (如果不存在的话)
        client.command(f'CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}')
        
        # 3. 定义建表语句
        # 我们需要根据 DataFrame 的列来精心设计表的结构
        create_table_query = f'''
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
        '''
        client.command(create_table_query)
        print(f"数据库 '{DATABASE_NAME}' 和表 '{TABLE_NAME}' 已准备就绪。")        
        # 4. 准备数据用于插入
        # a. 复制一份数据，避免修改原始的 DataFrame
        df_to_insert = df.copy()
        
        # b. 添加你最初要求的描述性字段
        df_to_insert['分析周期'] = 'hourly'
        
        # c. 把索引（时间）变回普通列，并重命名以匹配表结构
        df_to_insert = df_to_insert.reset_index()
        df_to_insert.rename(columns={'_time': '时间段', 
                                     '中位数 (Q2)': '中位数_Q2',
                                     '10th百分位数': 'P10'}, inplace=True)
        
        # d. 确保列的顺序和类型与表定义一致
        final_columns = [
            '时间段', '分析周期', '均值', '中位数_Q2', '最大值', '最小值', 
            'Q1', 'Q3', 'P10', '极差', '超过Q3占时比', '极差的时间变化率'
        ]
        df_to_insert = df_to_insert[final_columns]
        
        # 5. 插入数据
        print(f"正在插入 {len(df_to_insert)} 行数据...")
        client.insert_df(f'{DATABASE_NAME}.{TABLE_NAME}', df_to_insert)
        
        print("数据成功存入 ClickHouse！")

    except Exception as e:
        print(f"存入 ClickHouse 时发生错误: {e}")
def get_latest_timestamp_from_clickhouse() -> pd.Timestamp:
    """查询 ClickHouse，获取已有特征数据的最新时间戳。"""
    try:
        client = clickhouse_connect.get_client(host=CLICKHOUSE_HOST, port=CLICKHOUSE_PORT, username=CLICKHOUSE_USER, password=CLICKHOUSE_PASSWORD)
        tables = client.query_df(f"SHOW TABLES FROM {DATABASE_NAME}")
        if TABLE_NAME not in tables['name'].values:
            return None

        query = f"SELECT max(`时间段`) FROM {DATABASE_NAME}.{TABLE_NAME}"
        result = client.query(query)
        latest_time = result.first_row[0] if result.first_row else None
        
        if latest_time:
            return pd.to_datetime(latest_time).tz_localize('UTC')
        return None
    except Exception as e:
        print(f"查询 ClickHouse 最新时间戳时出错: {e}")
        return None
    

if __name__ == "__main__":
    last_processed_time = get_latest_timestamp_from_clickhouse()
    
    # 如果没有数据，我们就从2021年开始处理
    if last_processed_time is None:
        start_year = 2021
    else:
        # 否则，从上一次处理时间的年份开始
        start_year = last_processed_time.year
        
    # 我们处理到当前年份
    end_year = datetime.datetime.now().year

    # 2. 按年份循环，每年处理一次
    for year in range(start_year, end_year + 1):
        print(f"\n--- 正在处理年份: {year} ---")
        
        # a. 定义这一年的开始和结束时间 (UTC)
        year_start_utc_str = f"{year}-01-01T00:00:00Z"
        year_end_utc_str = f"{year+1}-01-01T00:00:00Z"
        
        # b. 提取这一年的原始数据
        yearly_df = get_humidity_data(start_time_utc_str=year_start_utc_str, stop_time_utc_str=year_end_utc_str)
        
        if yearly_df.empty:
            print(f"  年份 {year} 没有新的原始数据。")
            continue
            
        print(f"  成功提取了 {len(yearly_df)} 条 {year} 年的原始数据。")
        
        # c. 为这一年的数据计算特征
        print("  正在计算小时级特征...")
        hourly_features = calculate_hourly_features(yearly_df)
        
        # d. 过滤掉我们可能已经处理过的数据
        if last_processed_time and not hourly_features.empty:
            hourly_features = hourly_features[hourly_features.index > last_processed_time]

        # e. 将这一年的新特征存入 ClickHouse
        if not hourly_features.empty:
            print(f"  准备将 {len(hourly_features)} 条新的小时级特征存入 ClickHouse...")
            store_features_to_clickhouse(hourly_features)
        else:
            print("  该年份没有需要更新的特征数据。")