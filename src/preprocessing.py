import datetime

import pandas as pd

from src import config


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
