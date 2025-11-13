import warnings
from zoneinfo import ZoneInfo

import pandas as pd

from src.io import (
    get_clickhouse_client,
    get_full_table_from_clickhouse,
    get_global_time_range,
    store_dataframe_to_clickhouse,
)
from src.preprocessing import preprocess_limited_interpolation

warnings.simplefilter(action="ignore", category=FutureWarning)


def main():
    """
    从 ClickHouse 提取 -> 线性插值 -> 存入新 ClickHouse 表
    """

    SOURCE_DB = "original_data"
    SOURCE_TABLE = "sensor_temp_humidity"
    TARGET_DB = "original_data_interpolated"
    TARGET_TABLE = "sensor_temp_humidity_interpolated"
    LOCAL_TZ = ZoneInfo("Asia/Shanghai")

    try:
        shared_client = get_clickhouse_client(target="shared")
    except Exception as e:
        print(f"❌ 无法连接到 ClickHouse 服务器，流水线终止: {e}")
        return

    min_date_utc, max_date_utc = get_global_time_range(
        client=shared_client,
        database_name=SOURCE_DB,
        table_name=SOURCE_TABLE,
        time_column="time",  # 原始表里的时间列名
    )

    min_date_local = min_date_utc.tz_convert(LOCAL_TZ)
    max_date_local = max_date_utc.tz_convert(LOCAL_TZ)

    all_months = (
        pd.date_range(
            start=min_date_local.replace(day=1),  # 从最小日期的那个月1号开始
            end=max_date_local,
            freq="MS",  # MS = Month Start (每月开始)
        )
        .tz_localize(None)
        .tz_localize(LOCAL_TZ)
    )  # 确保时区正确

    # --- 按月份循环处理 ---
    for i, month_start in enumerate(all_months):
        # a. 计算当前月份的起止时间
        month_end = month_start + pd.DateOffset(months=1)

        # b. 提取 (Extract) - 只提取当月的数据
        raw_df_monthly = get_full_table_from_clickhouse(
            client=shared_client,
            database_name=SOURCE_DB,
            table_name=SOURCE_TABLE,
            start_time=month_start,  # 传入当前月的开始时间
            end_time=month_end,  # 传入当前月的结束时间
        )
        if raw_df_monthly.empty:
            print("  ► 该月没有数据，跳过。")
            continue
        if "temple_id" not in raw_df_monthly.columns:
            print(f"❌ 错误：源表 {SOURCE_DB}.{SOURCE_TABLE} 中缺少 'temple_id' 列。")
            print(f"   ► 找到的列: {raw_df_monthly.columns.tolist()}")
            break  # 终止整个脚本

        # print("DEBUG: DataFrame 的索引 (Index):", raw_df_monthly.index)
        # c. 预处理 (Preprocess) - 只对当月数据进行插值
        raw_df_monthly = raw_df_monthly.set_index("time")
        processed_devices = []
        for device_id, device_df in raw_df_monthly.groupby("device_id"):
            temple_id = device_df["temple_id"].iloc[0]
            device_df_resampled = device_df.resample("1min").mean(numeric_only=True)
            interpolated_df = preprocess_limited_interpolation(
                device_df_resampled,  # 传入重采样后的数据
                resample_freq="1min",
                start_time=month_start,
                end_time=month_end,
                gap_threshold_hours=2.0,
            )
            if not interpolated_df.empty:
                interpolated_df["device_id"] = device_id
                interpolated_df["temple_id"] = temple_id
                processed_devices.append(interpolated_df)
        if not processed_devices:
            print("   ► 该月所有设备数据处理后为空，跳过。")
            continue

        interpolated_df_monthly = pd.concat(processed_devices)
        interpolated_df_monthly = interpolated_df_monthly.reset_index().rename(
            columns={"_time": "time"}
        )
        # d. 存储 (Store) - 只存储当月的结果
        store_dataframe_to_clickhouse(
            df=interpolated_df_monthly,
            client=shared_client,
            database_name=TARGET_DB,
            table_name=TARGET_TABLE,
        )

    if "shared_client" in locals() and shared_client.connection:
        shared_client.disconnect()
        print("\n✅ ClickHouse 连接已关闭。")


if __name__ == "__main__":
    main()
