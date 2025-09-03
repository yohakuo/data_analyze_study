import pandas as pd
from influxdb_client import InfluxDBClient
import clickhouse_connect
import datetime
from zoneinfo import ZoneInfo

# --- 数据库连接配置 ---
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "study2025"
INFLUXDB_ORG = "task3"
INFLUXDB_BUCKET = "cave45"

CLICKHOUSE_HOST = 'localhost'
CLICKHOUSE_PORT = 8123
CLICKHOUSE_USER = 'default'
CLICKHOUSE_PASSWORD = 'study2025'
DATABASE_NAME = 'feature_db'

def get_humidity_data() -> pd.DataFrame:
    with InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG) as client:
        query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: 0)
          |> filter(fn: (r) => r["_measurement"] == "adata")
          |> filter(fn: (r) => r["_field"] == "空气湿度")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        print("正在从 InfluxDB 查询所有原始数据，请稍候...")
        df = client.query_api().query_data_frame(query=query, org=INFLUXDB_ORG)
        
        if not df.empty:
            df_cleaned = df[['_time', '空气湿度']]
            df_cleaned = df_cleaned.set_index('_time')
            df_cleaned.index = pd.to_datetime(df_cleaned.index)
            df_cleaned['空气湿度'] = pd.to_numeric(df_cleaned['空气湿度'], errors='coerce')
            df_cleaned = df_cleaned.dropna()
            return df_cleaned
        return pd.DataFrame()

def calculate_and_store_global_volatility(df: pd.DataFrame):
    """计算整体波动性特征，并将其存入 ClickHouse 的一个新表中。"""
    if df.empty:
        return

    print("\n正在计算整体波动性特征...")
    humidity_series = df['空气湿度']
    
    # 1. 计算所有需要的波动性特征
    features = {
        'calculation_time': datetime.datetime.utcnow(),
        'std_dev': humidity_series.std(),
        'mad_mean': (humidity_series - humidity_series.mean()).abs().mean(),
        'mad_median': (humidity_series - humidity_series.median()).abs().mean(),
        'cv': humidity_series.std() / humidity_series.mean(),
        'autocorr_lag1': humidity_series.autocorr(lag=1)
    }
    features['is_volatile_over_15pct'] = 1 if features['cv'] > 0.15 else 0

    print("--- 整体波动性分析结果 ---")
    for key, value in features.items():
        print(f"{key}: \t{value:.4f}" if isinstance(value, float) else f"{key}: \t{value}")
    
    # 2. 将计算结果（一个字典）转换为单行的 DataFrame，为插入做准备
    features_df = pd.DataFrame([features])
    
    # 3. 连接 ClickHouse 并定义一个新的表名
    TABLE_NAME = 'humidity_global_features' 
    client = clickhouse_connect.get_client(host=CLICKHOUSE_HOST, port=CLICKHOUSE_PORT, username=CLICKHOUSE_USER, password=CLICKHOUSE_PASSWORD)
    client.command(f'CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}')

    # 4. 为这个新表定义它的结构
    create_table_query = f'''
    CREATE TABLE IF NOT EXISTS {DATABASE_NAME}.{TABLE_NAME}
    (
        `calculation_time` DateTime,
        `std_dev` Float64,
        `mad_mean` Float64,
        `mad_median` Float64,
        `cv` Float64,
        `is_volatile_over_15pct` UInt8,
        `autocorr_lag1` Float64
    )
    ENGINE = MergeTree()
    ORDER BY calculation_time
    '''
    client.command(create_table_query)
    print(f"\nClickHouse 的新表 '{TABLE_NAME}' 已准备就绪。")

    # 5. 插入我们刚刚计算出的那【一行】整体特征数据
    client.insert_df(f'{DATABASE_NAME}.{TABLE_NAME}', features_df)
    print(f"✅ 成功！整体波动性特征已存入新表 '{TABLE_NAME}'。")


if __name__ == "__main__":
    raw_data_df = get_humidity_data()
    if not raw_data_df.empty:
        calculate_and_store_global_volatility(raw_data_df)