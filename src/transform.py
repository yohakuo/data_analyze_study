import datetime

import pandas as pd

from src import config
from src.features.statistica import FEATURE_CALCULATORS
from src.features.statistica import calculate_features as _calculate_single_field_features


def preprocess_timeseries_data(
    df: pd.DataFrame,
    resample_freq: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> pd.DataFrame:
    """
    对原始时间序列数据进行标准化预处理：重采样、插值、时区转换。
    """
    if df.empty:
        return df

    # 1. 统一时区为 UTC 进行处理
    df = df.tz_localize("UTC") if df.index.tz is None else df.tz_convert("UTC")

    # 2. 创建完整时间轴并重采样
    start_utc = start_time.astimezone(datetime.timezone.utc)
    end_utc = end_time.astimezone(datetime.timezone.utc)
    full_range = pd.date_range(start=start_utc, end=end_utc, freq=resample_freq)
    reindexed_df = df.reindex(full_range)

    # 3. 插值填充：先线性插值，再用前后值填充边缘
    filled_df = reindexed_df.interpolate(method="linear").ffill().bfill()

    # 4. 转换回本地时区
    final_df = filled_df.tz_convert(config.LOCAL_TIMEZONE)
    final_df.index.name = "_time"

    print(f"  ► 预处理完成。处理后的数据共有 {len(final_df)} 条记录。")
    return final_df


def preprocess_limited_interpolation(
    df: pd.DataFrame,
    resample_freq: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    gap_threshold_hours: float = 2.0,
) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.tz_localize("UTC") if df.index.tz is None else df.tz_convert("UTC")

    start_utc = start_time.astimezone(datetime.timezone.utc)
    # end_time 应该是 "下一个月的第一分钟" 之前，
    # 否则 date_range(..., end=...) 会漏掉 23:59:00
    end_utc = end_time.astimezone(datetime.timezone.utc)
    # 创建一个完整的、从月初到月末的 1 分钟索引
    full_range = pd.date_range(start=start_utc, end=end_utc, freq=resample_freq, inclusive="left")
    reindexed_df = df.reindex(full_range)

    try:
        threshold_duration = pd.Timedelta(hours=gap_threshold_hours)
        resample_duration = pd.Timedelta(resample_freq)
    except ValueError:
        raise ValueError(
            f"无效的 resample_freq: '{resample_freq}'。请使用 'T', 'min', 'H' 等 pandas 频率字符串。"
        )

    if resample_duration.total_seconds() <= 0:
        raise ValueError(f"resample_freq '{resample_freq}' 必须是正的时间间隔。")

    # 计算插值的最大连续点数
    limit_count = int(threshold_duration / resample_duration) - 1
    if limit_count < 1:
        print(
            f"警告: resample_freq ({resample_freq}) 大于或等于门限 ({gap_threshold_hours}h)。将不进行插值。"
        )
        limit_count = 0

    interpolated_df = reindexed_df.interpolate(method="linear", limit=limit_count)

    final_df = interpolated_df.dropna()
    final_df = final_df.tz_convert(config.LOCAL_TIMEZONE)
    final_df.index.name = "_time"

    print(f"  ► 预处理(有限插值)完成。处理后的数据共有 {len(final_df)} 条记录。")
    return final_df


def transform_device_data(
    device_df: pd.DataFrame, fields_to_process: list, features_to_calc: list, freq: str
) -> pd.DataFrame:
    """
    [T] 转换包装器：
    接收【单个设备】的数据，为【多个字段】计算【多个特征】。
    """
    if device_df.empty:
        return pd.DataFrame()

    all_features_list = []

    try:
        device_df_indexed = device_df.set_index("time")

        # 同时提取 device_id 和 temple_id
        device_id = device_df["device_id"].iloc[0]
        if "temple_id" not in device_df.columns:
            print("❌ [T] 转换失败：数据中缺少 'temple_id' 列。")
            return pd.DataFrame()
        temple_id = device_df["temple_id"].iloc[0]
    except KeyError as e:
        print(f"❌ [T] 转换失败：数据中缺少 'time' 或 'device_id' 列。{e}")
        return pd.DataFrame()
    # 循环处理每个【字段】(e.g., 'humidity', 'temperature')
    for field_name in fields_to_process:
        # -----------------------------------------------------------------
        # 2. 调用 (Call)
        # -----------------------------------------------------------------
        # ‼️ 在这里，【调用】了从 statistica.py 导入的函数
        # -----------------------------------------------------------------
        wide_df = _calculate_single_field_features(
            device_df_indexed, field_name=field_name, feature_list=features_to_calc, freq=freq
        )

        if wide_df.empty:
            continue

        # ... (后续的 Melt 和元数据添加) ...
        long_df = wide_df.reset_index().melt(
            id_vars=["time"], var_name="feature_key", value_name="value"
        )
        long_df["device_id"] = device_id
        long_df["field_name"] = field_name
        long_df["temple_id"] = temple_id
        all_features_list.append(long_df)

    if not all_features_list:
        return pd.DataFrame()

    final_df = pd.concat(all_features_list)
    # 解决入库后时间后移8小时的问题
    final_df = final_df.reset_index()
    final_df["time"] = final_df["time"].dt.tz_localize(None)

    return final_df.dropna(subset=["value"])
