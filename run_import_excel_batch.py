import datetime
import os
import re
import warnings

from clickhouse_driver import Client
import pandas as pd

from src import config
from src.dataset import _create_raw_tables_if_not_exists, get_clickhouse_client, process_excel_file

warnings.filterwarnings(
    "ignore", category=UserWarning, message="Workbook contains no default style"
)

DATA_DIRECTORY = r"C:\Users\23947\Desktop\todo\第一批数据(27个窟-2023~2024)\4"


def main():
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"任务开始时间: {formatted_time}")
    try:
        DB = config.CLICKHOUSE_SHARED_DB
        client = get_clickhouse_client(DB)
    except Exception:
        print("无法连接到 ClickHouse，脚本终止。")
        return

    _create_raw_tables_if_not_exists(client)

    if not os.path.isdir(DATA_DIRECTORY):
        print(f"错误：数据目录 '{DATA_DIRECTORY}' 不存在。")
        return

    for filename in os.listdir(DATA_DIRECTORY):
        if filename.endswith((".xlsx", ".xls")):
            file_path = os.path.join(DATA_DIRECTORY, filename)
            try:
                process_excel_file(
                    file_path,
                    client,
                    config.RAW_SENSOR_MAPPING_CONFIG,
                    config.RAW_FILE_PARSING_CONFIG,
                )
            except Exception as e:
                print(f"处理文件 {filename} 时发生意外错误: {e}")

    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"任务完成时间: {formatted_time}")
    client.disconnect()


if __name__ == "__main__":
    main()
