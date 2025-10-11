import datetime
import os
from zoneinfo import ZoneInfo

from src import config
from src.dataset import get_timeseries_data, preprocess_timeseries_data


def main():
    start_time = datetime.datetime(2021, 1, 1, 0, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    end_time = datetime.datetime(2025, 7, 10, 15, 16, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    raw_df = get_timeseries_data(
        measurement_name=config.MEASUREMENT_NAME,
        field_name=config.FIELD_NAME,  # '空气湿度（%）'
        start_time=start_time,
        stop_time=end_time,
    )

    if raw_df.empty:
        print("指定范围内无数据。")
        return

    # print("\n--- [对比] 预处理之前 ---")
    # print(f"原始数据共有 {len(raw_df)} 行。")
    # print("前5行预览:")
    # print(raw_df.head())

    preprocessed_df = preprocess_timeseries_data(
        raw_df, "1T", start_time=start_time, end_time=end_time
    )

    print("\n--- 预处理之后 ---")
    print(f"预处理后的数据共有 {len(preprocessed_df)} 行。")
    print("前5行预览:")
    print(preprocessed_df.head())

    output_path = os.path.join(config.PROCESSED_DATA_PATH, "preprocessed_data.parquet")
    os.makedirs(config.PROCESSED_DATA_PATH, exist_ok=True)
    preprocessed_df.to_parquet(output_path)
    print("✅ 保存成功。")


if __name__ == "__main__":
    main()
