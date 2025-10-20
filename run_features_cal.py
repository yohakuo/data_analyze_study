import datetime
import time
from zoneinfo import ZoneInfo

import pandas as pd

from src.dataset import (
    get_clickhouse_client,
    get_timeseries_data,
    store_features_to_center_clickhouse,
    store_features_to_clickhouse,
)
from src.features import calculate_features

# 查询数据--get_timeseries_data
MEASUREMENT_NAME = "DongNan"
FIELD_NAME = "空气湿度（%）"  # "空气湿度（%）"、"空气温度（℃）"
ANALYSIS_START_TIME_LOCAL = datetime.datetime(2022, 1, 4, tzinfo=ZoneInfo("Asia/Shanghai"))
ANALYSIS_STOP_TIME_LOCAL = datetime.datetime(2022, 1, 14, tzinfo=ZoneInfo("Asia/Shanghai"))
# 存储--store_features_to_clickhouse
FEATURES_TABLE = "features_caculate_dn"  # "features_caculate"
# 元数据字段
TEMPLE_ID = "045"
DEVICE_ID = "201A"
FREQ = "h"  # h\D\W\M
STATS_CYCLE = "hour"
FEATURE_KEY = "mean"


def main():
    # start_time = time.perf_counter()
    # TARGET_DATABASE = "shared"  #  'local'
    # db_client=get_clickhouse_client(target=TARGET_DATABASE)\\

    # raw_df = get_timeseries_data(
    #     measurement_name=MEASUREMENT_NAME,
    #     field_name=FIELD_NAME,
    #     start_time=None,  # ANALYSIS_START_TIME_LOCAL,
    #     stop_time=None,  # ANALYSIS_STOP_TIME_LOCAL,
    # )

    # if raw_df.empty:
    #     print("没有提取到数据，流程结束。")
    #     return

    DATA_FILENAME = "D:\\Projects\\ts_fe\\data\\processed\\preprocessed_data.parquet"

    try:
        raw_df = pd.read_parquet(DATA_FILENAME)
        print("加载成功！")
    except FileNotFoundError:
        print(f"错误：找不到文件 '{DATA_FILENAME}'。")
        exit()

    FEATURE_LIST = ["均值"]
    humidity_features_wide = calculate_features(
        raw_df, field_name=FIELD_NAME, feature_list=FEATURE_LIST, freq=FREQ
    )

    store_features_to_clickhouse(
        df=humidity_features_wide,
        table_name=FEATURES_TABLE,
        field_name=FIELD_NAME,
        device_id=DEVICE_ID,
        temple_id=TEMPLE_ID,
        stats_cycle=STATS_CYCLE,
    )
    # end_time = time.perf_counter()
    # print(f"\n🎉 ** 总耗时: {end_time - start_time:.2f} 秒 ** 🎉")


if __name__ == "__main__":
    main()
