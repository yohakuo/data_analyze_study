import datetime
import os
from zoneinfo import ZoneInfo

from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters

from src import config
from src.io import get_timeseries_data


def main():
    start_time = datetime.datetime(2021, 1, 1, tzinfo=ZoneInfo("Asia/Shanghai"))
    end_time = datetime.datetime(2021, 1, 2, tzinfo=ZoneInfo("Asia/Shanghai"))

    raw_df = get_timeseries_data(
        measurement_name=config.INFLUX_MEASUREMENT_NAME,
        field_name=config.FIELD_NAME,  # '空气湿度（%）'
        start_time=start_time,
        stop_time=end_time,
    )
    if raw_df.empty:
        return
    print(f"成功提取 {len(raw_df)} 条原始记录。")

    # 将数据转换为 tsfresh 需要的“长”格式.将时间索引变回普通列
    df_for_tsfresh = raw_df.reset_index()
    df_for_tsfresh["id"] = "dongnan_humidity"
    df_for_tsfresh.rename(columns={"_time": "time", config.FIELD_NAME: "value"}, inplace=True)

    # 计算所有可能的特征
    settings = ComprehensiveFCParameters()
    extracted_features = extract_features(
        df_for_tsfresh,
        column_id="id",
        column_sort="time",
        default_fc_parameters=settings,
        n_jobs=2,  # 使用2个CPU核心并行计算，可以加速
    )
    print(extracted_features.head())
    output_filename = os.path.join(config.PROCESSED_DATA_PATH, "tsfresh_features.csv")
    extracted_features.to_csv(output_filename)


if __name__ == "__main__":
    main()
