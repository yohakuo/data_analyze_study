import pandas as pd
from influxdb_client import InfluxDBClient
import clickhouse_connect
import datetime
import csv
from zoneinfo import ZoneInfo 
import random

pd.set_option('display.max_rows', None)

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
TABLE_NAME = 'humidity_hourly_features'


def get_adjacent_feature_rows_from_clickhouse():
    """从ClickHouse随机获取【相邻的两行】特征数据。"""
    print("--- 步骤 1: 正在从 ClickHouse 随机抽取相邻的两条特征数据... ---")
    try:
        client = clickhouse_connect.get_client(
            host=CLICKHOUSE_HOST, port=CLICKHOUSE_PORT, 
            username=CLICKHOUSE_USER, password=CLICKHOUSE_PASSWORD
        )
        
        # a. 先获取总行数，以确定随机抽样的范围
        count_query = f"SELECT count() FROM {DATABASE_NAME}.{TABLE_NAME}"
        total_rows = client.query(count_query).first_row[0]
        
        if total_rows < 2:
            print("错误：表中的数据不足两行，无法抽取相邻数据。")
            return None
            
        # b. 生成一个安全的随机“起点”(offset)
        # 范围是 0 到 (总行数 - 2)，确保我们总能抽到两行
        random_offset = random.randint(0, total_rows - 2)
        
        # c. 使用 LIMIT 2 OFFSET ... 语法来抽取相邻的两行
        query = f"SELECT * FROM {DATABASE_NAME}.{TABLE_NAME} ORDER BY `时间段` ASC LIMIT 2 OFFSET {random_offset}"
        
        adjacent_rows_df = client.query_df(query)
            
        return adjacent_rows_df
    except Exception as e:
        print(f"连接或查询 ClickHouse 时出错: {e}")
        return None

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


if __name__ == "__main__":
    pd.set_option('display.max_rows', None) # 确保打印所有特征
    LOCAL_TIMEZONE = ZoneInfo("Asia/Shanghai")

    # 模块一：获取标准答案 (现在是两行)
    standard_answers = get_adjacent_feature_rows_from_clickhouse()
    
    if standard_answers is not None and len(standard_answers) == 2:
        
        previous_hour = standard_answers.iloc[0]
        current_hour = standard_answers.iloc[1]

        # 为了显示，我们转换时间
        prev_time_local = pd.to_datetime(previous_hour['时间段']).tz_localize('UTC').tz_convert(LOCAL_TIMEZONE)
        curr_time_local = pd.to_datetime(current_hour['时间段']).tz_localize('UTC').tz_convert(LOCAL_TIMEZONE)
        
        print("✅ 成功获取！将对下面“当前小时”的数据进行验证：")
        print("\n--- 当前小时特征(程序计算) ---")
        print(current_hour)
        
        # 模块二：获取“当前小时”的原始数据用于验证
        timestamp_utc = pd.to_datetime(current_hour['时间段']).tz_localize('UTC')
        raw_data = get_raw_data_for_hour(timestamp_utc)
        
        if raw_data:
            filename = f"validation_data_{curr_time_local.strftime('%Y-%m-%d_%H%M')}.csv"
            
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['空气湿度'])
                for value in raw_data:
                    writer.writerow([value])
            
            
            print("\n" + "="*50)
            print("      *** 高级特征手动验证 ***")
            print("="*50)
            
            # 从抽取的两行数据中获取极差
            prev_range = previous_hour['极差']
            curr_range = current_hour['极差']
            
            print(f"前一小时的极差: {prev_range}")
            print(f"当前小时的极差: {curr_range}")
            
            # 手动计算变化率
            manual_rate_of_change = 0.0
            if prev_range != 0: # 避免除以零的错误
                manual_rate_of_change = (curr_range - prev_range) / prev_range
            
            print(f"手动计算的变化率: ({curr_range} - {prev_range}) / {prev_range} = {manual_rate_of_change:.4f}")
            script_rate_of_change = current_hour['极差的时间变化率']
            print(f"脚本计算的变化率: {script_rate_of_change:.4f}")
        else:
            print("未能从 InfluxDB 获取到这个小时的原始数据。")