import datetime
import os
from zoneinfo import ZoneInfo

from src import config
from src.dataset import (
    get_timeseries_data,
    preprocess_limited_interpolation,
    preprocess_timeseries_data,
    store_dataframe_to_clickhouse,
)


def main():
    start_time = datetime.datetime(2021, 1, 1, 0, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    end_time = datetime.datetime(2025, 7, 10, 15, 16, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    # 表名
    target_table_name = "TH_preprocessed"
    raw_df = get_timeseries_data(
        measurement_name=config.MEASUREMENT_NAME,
        field_name=config.FIELD_NAME,
        start_time=start_time,
        stop_time=end_time,
    )
    if raw_df.empty:
        return

    # preprocessed_df = preprocess_timeseries_data(
    #     raw_df, "1T", start_time=start_time, end_time=end_time
    # )
    preprocessed_df = preprocess_limited_interpolation(raw_df, "1T", start_time, end_time)

    # output_path = os.path.join(config.PROCESSED_DATA_PATH, "preprocessed_data.parquet")
    # os.makedirs(config.PROCESSED_DATA_PATH, exist_ok=True)
    # preprocessed_df.to_parquet(output_path)
    # print("✅ 保存成功。")

    ## 保存到中心服务器
    store_dataframe_to_clickhouse(
        df=preprocessed_df,
        table_name=target_table_name,
        target="shared",
        chunk_size=50000,
    )


if __name__ == "__main__":
    main()
