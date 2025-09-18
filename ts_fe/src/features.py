# 所有核心的数据计算和转换逻辑,位于dataset之上
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import detrend

from src import dataset


# 统计特征计算
def calculate_percent_above_q3(series):
    if len(series) < 4:
        return 0.0
    q3 = series.quantile(0.75)
    # 避免 Q3 等于最大值时，没有数能大于 Q3 的情况
    if q3 == series.max():
        return 0.0
    count_above_q3 = (series > q3).sum()
    percent_above_q3 = count_above_q3 / len(series)
    return percent_above_q3


def calculate_hourly_features(df: pd.DataFrame, field_name: str) -> pd.DataFrame:
    if df.empty or field_name not in df.columns:
        print(f"错误：数据为空或找不到指定的字段 '{field_name}'")
        return pd.DataFrame()

    print(f"\n正在为字段 '{field_name}' 计算小时级特征...")

    # 1. 先选择要分析的列，并按小时重采样
    resampled_data = df[field_name].resample("H")

    # 2. 然后使用 new_column_name='function' 的语法进行聚合
    hourly_stats = resampled_data.agg(
        均值="mean",
        中位数="median",
        最大值="max",
        最小值="min",
        Q1=lambda x: x.quantile(0.25),
        Q3=lambda x: x.quantile(0.75),
        P10=lambda x: x.quantile(0.10),
    )

    hourly_stats["极差"] = hourly_stats["最大值"] - hourly_stats["最小值"]

    percent_above_q3 = df[field_name].resample("H").apply(calculate_percent_above_q3)
    hourly_stats["超过Q3占时比"] = percent_above_q3

    hourly_stats["极差的时间变化率"] = hourly_stats["极差"].pct_change().fillna(0)

    hourly_stats.rename(columns={"中位数": "中位数 (Q2)", "P10": "10th百分位数"}, inplace=True)

    print(f"字段 '{field_name}' 的小时级特征计算完成！")
    return hourly_stats


# 波动性特征计算
def mad_from_mean(series):
    """计算对均值的平均绝对偏差"""
    return (series - series.mean()).abs().mean()


def mad_from_median(series):
    """计算对中位数的平均绝对偏差"""
    return (series - series.median()).abs().mean()


def autocorr_lag1(series):
    """计算一阶自相关系数"""
    if len(series) < 2:
        return None
    return series.autocorr(lag=1)


def calculate_daily_volatility(df: pd.DataFrame, field_name: str) -> pd.DataFrame:
    """
    按天('D')重采样，计算每日波动性特征。
    """
    if df.empty or field_name not in df.columns:
        return pd.DataFrame()

    print(f"\n正在为字段 '{field_name}' 计算每日波动性特征...")

    daily_stats = (
        df[field_name]
        .resample("D")
        .agg(
            均值="mean",
            标准差="std",
            平均绝对偏差_均值=mad_from_mean,
            平均绝对偏差_中位数=mad_from_median,
            一阶自相关=autocorr_lag1,
        )
    )

    daily_stats["变异系数"] = daily_stats["标准差"] / daily_stats["均值"]

    daily_stats.dropna(inplace=True)

    print("每日波动性计算完成！")
    return daily_stats


def find_longest_stable_period(df: pd.DataFrame, threshold: float):
    """
    在每日波动性数据中，寻找最长的连续平稳期。
    """
    df["is_stable"] = df["标准差"] < threshold

    # 使用 groupby 和 shift() 的技巧来找出所有连续的平稳期
    streaks = (
        df[df["is_stable"]].groupby((df["is_stable"] != df["is_stable"].shift()).cumsum()).size()
    )

    if streaks.empty:
        return None, None, 0

    longest_streak_len = streaks.max()
    # 找到最长连胜的组ID
    streak_group_id = streaks.idxmax()

    # 从原始DataFrame中，根据ID找到对应的日期
    streak_df = df[
        (df["is_stable"])
        & ((df["is_stable"] != df["is_stable"].shift()).cumsum() == streak_group_id)
    ]

    start_date = streak_df.index.min().date()
    end_date = streak_df.index.max().date()

    return start_date, end_date, longest_streak_len


# FFT分析
def analyze_with_fft(df: pd.DataFrame, field_name: str):
    """
    对 DataFrame 中的指定字段进行FFT分析，并找出最主要的周期
    """
    if df.empty:
        return None

    print(f"\n正在对字段 '{field_name}' 进行频谱分析 (FFT)...")

    # 1. 去除趋势
    print("正在去除数据的长期趋势...")
    detrended_values = detrend(df[field_name].values)

    # 2. FFT 计算
    N = len(detrended_values)
    T = (df.index[1] - df.index[0]).total_seconds()  # 自动计算采样间隔
    yf = fft(detrended_values)
    xf = fftfreq(N, T)[: N // 2]
    amplitude = 2.0 / N * np.abs(yf[0 : N // 2])

    # 3. 将频率转换为小时为单位的周期
    periods_in_hours = np.full_like(xf, np.inf)
    non_zero_indices = xf > 0
    periods_in_hours[non_zero_indices] = 1 / xf[non_zero_indices] / 3600

    # 4. 组合成 DataFrame 并返回
    spectrum_df = pd.DataFrame({"周期(小时)": periods_in_hours, "强度(幅度)": amplitude})
    return spectrum_df
