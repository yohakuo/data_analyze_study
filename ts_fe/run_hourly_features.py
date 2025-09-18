import datetime
from zoneinfo import ZoneInfo

from src.dataset import get_timeseries_data, store_features_to_clickhouse
from src.features import calculate_hourly_features

# 查询数据--get_timeseries_data
MEASUREMENT_NAME = "DongNan"
FIELD_NAME = "空气湿度（%）"  # "空气湿度（%）"、"空气温度（℃）"
ANALYSIS_START_TIME_LOCAL = datetime.datetime(2022, 1, 4, tzinfo=ZoneInfo("Asia/Shanghai"))
ANALYSIS_STOP_TIME_LOCAL = datetime.datetime(2022, 1, 14, tzinfo=ZoneInfo("Asia/Shanghai"))

# 存储--store_features_to_clickhouse
HOURLY_FEATURES_TABLE = "features_calulate"

## 元数据字段
TEMPLE_ID = "045"
DEVICE_ID = "201A"
# stats_start_time = ANALYSIS_START_TIME_LOCAL
# monitored_variable = FIELD_NAME
STATS_CYCLE = "hour"
FEATURE_KEY = "mean"
# feature_value = 计算结果
# standby_field01 备用字段
# created_at  记录创建时间


def main():
    raw_df = get_timeseries_data(
        measurement_name=MEASUREMENT_NAME,
        field_name=FIELD_NAME,
        start_time=ANALYSIS_START_TIME_LOCAL,
        stop_time=ANALYSIS_STOP_TIME_LOCAL,
    )
    if raw_df.empty:
        print("没有提取到数据，流程结束。")
        return

    humidity_features_wide = calculate_hourly_features(
        raw_df, field_name=FIELD_NAME, feature_key=FEATURE_KEY
    )

    store_features_to_clickhouse(
        df=humidity_features_wide,
        table_name=HOURLY_FEATURES_TABLE,
        field_name=FIELD_NAME,
        device_id=DEVICE_ID,
        temple_id=TEMPLE_ID,
        stats_cycle=STATS_CYCLE,
    )


if __name__ == "__main__":
    main()
