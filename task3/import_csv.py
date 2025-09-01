import csv
from datetime import datetime
from influxdb_client import InfluxDBClient, Point, WriteOptions
from zoneinfo import ZoneInfo

# --- 1. 基本配置  ---
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "study2025" 
INFLUXDB_ORG = "task3"
INFLUXDB_BUCKET = "cave45"

CSV_FILE_PATH = 'real_data.CSV'   
MEASUREMENT_NAME = 'adata'    
LOCAL_TIMEZONE = "Asia/Shanghai"

# --- 2. CSV 列配置  ---
# 时间列的列名
TIMESTAMP_COLUMN = '采集时间' 
TIMESTAMP_FORMAT = '%Y/%m/%d %H:%M:%S'

# Field 列 
FIELD_COLUMNS = ['空气温度', '空气湿度'] 

# Tag 列 (用于索引和分类的列)
TAG_COLUMNS = []


# --- 3. 脚本核心逻辑  ---
def main():
    """主函数，执行数据导入流程"""
    # 使用 WriteOptions 进行批量写入，这是最高效的方式
    with InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG) as client:
        with client.write_api(write_options=WriteOptions(batch_size=2000, flush_interval=10_000)) as write_api:
            
            print(f"开始从 CSV '{CSV_FILE_PATH}' 读取数据并导入 InfluxDB...")
            count = 0
            local_tz = ZoneInfo(LOCAL_TIMEZONE)

            try:
                with open(CSV_FILE_PATH, 'r', encoding='gbk') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        try:
                            # 创建一个数据点 (Point)
                            point = Point(MEASUREMENT_NAME)
                            
                            # 1. 添加 Tag
                            for tag_key in TAG_COLUMNS:
                                point.tag(tag_key, row[tag_key])
                                
                            # 2. 添加 Field，并尝试将值转为浮点数
                            for field_key in FIELD_COLUMNS:
                                try:
                                    field_value = float(row[field_key])
                                    point.field(field_key, field_value)
                                except (ValueError, TypeError):
                                    # 如果转换失败，就按原样存为字符串
                                    point.field(field_key, row[field_key])
                            
                            # 3. 添加时间戳
                            naive_dt = datetime.strptime(row[TIMESTAMP_COLUMN], TIMESTAMP_FORMAT)
                            aware_dt = naive_dt.replace(tzinfo=local_tz)
                            point.time(aware_dt)

                            # 将数据点写入API的缓冲区
                            write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
                            count += 1
                            if count % 2000 == 0:
                                print(f"已处理 {count} 行...")

                        except Exception as e:
                            print(f"处理第 {count + 1} 行时出错: {row}. 错误: {e}")
                
                print(f"\nCSV 文件读取完毕。等待数据全部写入 InfluxDB...")
            
            except FileNotFoundError:
                print(f"错误：找不到文件 '{CSV_FILE_PATH}'。请检查文件路径是否正确。")
                return # 找不到文件，直接退出
            except Exception as e:
                print(f"读取 CSV 文件时发生未知错误: {e}")
                return

    print(f"导入成功！总共处理了 {count} 行数据。")


if __name__ == "__main__":
    main()