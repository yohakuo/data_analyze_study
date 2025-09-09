import csv
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS
from src import config


def import_csv_to_influx(csv_filepath: str):
    """
    读取指定的 CSV 文件，并将其内容导入到 InfluxDB。

    Args:
        csv_filepath (str): 需要导入的 CSV 文件的完整路径。
    """
    print(f"▶️ 开始处理文件: {csv_filepath}")

    # 将配置项从 config 模块中读入
    local_tz = ZoneInfo(config.LOCAL_TIMEZONE)

    # 建立 InfluxDB 连接
    with InfluxDBClient(
        url=config.INFLUXDB_URL, token=config.INFLUXDB_TOKEN, org=config.INFLUXDB_ORG
    ) as client:
        # 使用同步(SYNCHRONOUS)写入选项，确保数据成功写入后才继续
        write_api = client.write_api(write_options=SYNCHRONOUS)

        points_buffer = []
        batch_size = 5000  # 定义批处理大小
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
                        print(
                            f"    处理文件 {csv_filepath} 的第 {total_count + 1} 行时出错: {e}"
                        )

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
            print(f"❌ 错误：找不到文件 '{csv_filepath}'。")
        except Exception as e:
            print(f"❌ 读取或写入文件 '{csv_filepath}' 时发生严重错误: {e}")
