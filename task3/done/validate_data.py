import pandas as pd
from influxdb_client import InfluxDBClient
import clickhouse_connect
import datetime
import csv
from zoneinfo import ZoneInfo #

pd.set_option('display.max_rows', None)

# --- 1. 数据库连接配置 ---
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "study2025"
INFLUXDB_ORG = "task3"
INFLUXDB_BUCKET = "cave45"

CLICKHOUSE_HOST = 'localhost'
CLICKHOUSE_PORT = 8123
CLICKHOUSE_USER = 'default'
CLICKHOUSE_PASSWORD = 'study2025'
DATABASE_NAME = 'feature_db'
TABLE_NAME = 'humidity_hourly_features'

# --- 2. 核心功能函数保持不变 ---
def get_random_feature_from_clickhouse():
    print("--- 正在从 ClickHouse 随机抽取一条特征数据... ---")
    try:
        client = clickhouse_connect.get_client(
            host=CLICKHOUSE_HOST, port=CLICKHOUSE_PORT, 
            username=CLICKHOUSE_USER, password=CLICKHOUSE_PASSWORD
        )
        query = f"SELECT * FROM {DATABASE_NAME}.{TABLE_NAME} ORDER BY rand() LIMIT 1"
        random_row_df = client.query_df(query)
        
        if random_row_df.empty:
            print("错误：无法从 ClickHouse 中获取数据，请确保表不为空。")
            return None
            
        return random_row_df.iloc[0]
    except Exception as e:
        print(f"连接或查询 ClickHouse 时出错: {e}")
        return None

def get_raw_data_for_hour(start_time_utc):
    end_time_utc = start_time_utc + datetime.timedelta(hours=1)
    
    start_utc_str = start_time_utc.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    end_utc_str = end_time_utc.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

    print(f"\n--- 正在从 InfluxDB 精准提取 {start_time_utc} (UTC) 的原始数据... ---")
    
    with InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG) as client:
        query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: {start_utc_str}, stop: {end_utc_str})
          |> filter(fn: (r) => r["_measurement"] == "adata")
          |> filter(fn: (r) => r["_field"] == "空气湿度")
          |> keep(columns: ["_value"])
        '''
        result = client.query_api().query(query=query, org=INFLUXDB_ORG)
        
        raw_data_list = [record.get_value() for table in result for record in table.records]
        return raw_data_list


# --- 3. 主执行逻辑 (全新升级版，专注时区转换和显示) ---
if __name__ == "__main__":
    # 定义我们关心的本地时区
    LOCAL_TIMEZONE = ZoneInfo("Asia/Shanghai")

    standard_answer = get_random_feature_from_clickhouse()
    
    if standard_answer is not None:
        # 1. 从 ClickHouse 获取的时间戳，强制其为 UTC 时间
        timestamp_utc = pd.to_datetime(standard_answer['时间段']).tz_localize('UTC')
        # 2. 创建一个转换后的“本地时间”版本，专门用于显示
        timestamp_local = timestamp_utc.tz_convert(LOCAL_TIMEZONE)

        print("✅ 成功获取程序计算的特征")

        # 3. 打印标准答案时，也显示转换后的本地时间
        display_answer = standard_answer.copy()
        display_answer['时间段'] = timestamp_local.strftime('%Y-%m-%d %H:%M:%S') # 格式化成易读的字符串
        print(display_answer)
        
        # 获取原始数据
        # **重要**：传递给后台函数的，依然是精确的、用于机器间对话的 UTC 时间
        raw_data = get_raw_data_for_hour(timestamp_utc)
        
        if raw_data:
            # 4. 创建 CSV 文件名时，使用我们熟悉的本地时间
            filename = f"{timestamp_local.strftime('%Y-%m-%d_%H%M')}_validation_data.csv"
            print(f"--- 正在将原始数据导出到 CSV 文件... ---")
            
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['空气湿度'])
                for value in raw_data:
                    writer.writerow([value])
            
            print(f"成功！原始数据已保存到文件: {filename}")
        else:
            print("未能从 InfluxDB 获取到这个小时的原始数据。")