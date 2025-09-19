# 所有核心的数据计算和转换逻辑,位于dataset之上
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import detrend

from src.utils import timing_decorator


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


FEATURE_CALCULATORS = {
    "均值": "mean",
    "中位数": "median",
    "最大值": "max",
    "最小值": "min",
    "标准差": "std",
    "Q1": lambda x: x.quantile(0.25),
    "Q3": lambda x: x.quantile(0.75),
    "P10": lambda x: x.quantile(0.10),
    "超过Q3占时比": calculate_percent_above_q3,
}


@timing_decorator
def calculate_hourly_features(
    df: pd.DataFrame, field_name: str, feature_list: list
) -> pd.DataFrame:
    """
    接收原始数据 DataFrame，并根据【指定的特征列表】，计算小时级统计特征。
    """
    if df.empty or field_name not in df.columns:
        print(f"错误：数据为空或找不到指定的字段 '{field_name}'")
        return pd.DataFrame()

    print(f"\n正在为字段 '{field_name}' 计算指定的 {len(feature_list)} 个小时级特征...")

    # a. 从我们的“大菜单”中，只挑出这次“点菜单”上有的菜
    agg_functions = {
        feature: FEATURE_CALCULATORS[feature]
        for feature in feature_list
        if feature in FEATURE_CALCULATORS
    }

    if not agg_functions:
        print("错误：指定的特征都不在可计算的列表中。")
        return pd.DataFrame()

    # b. 使用 .agg() 一次性计算所有被选中的特征
    hourly_stats = df[field_name].resample("h").agg(**agg_functions)

    # c. 计算依赖于其他结果的衍生特征
    #    只有当“最大值”和“最小值”都被点了，我们才能算“极差”
    if "最大值" in hourly_stats.columns and "最小值" in hourly_stats.columns:
        hourly_stats["极差"] = hourly_stats["最大值"] - hourly_stats["最小值"]

    #    只有当“极差”被计算出来了，我们才能算“极差的时间变化率”
    if "极差" in hourly_stats.columns:
        hourly_stats["极差的时间变化率"] = hourly_stats["极差"].pct_change().fillna(0)

    # d. 重命名，让列名更规范
    #    只有当“中位数”被点了，我们才需要重命名它
    if "中位数" in hourly_stats.columns:
        hourly_stats.rename(columns={"中位数": "中位数 (Q2)"}, inplace=True)

    print("指定的特征计算完成！")
    return hourly_stats


def recalculate_single_hour_features(raw_data_series: pd.Series) -> pd.Series:
    """
    接收一个小时的原始数据序列(Series)，重新计算所有（非变化率）特征并返回一个 Series。
    这个函数是自动化测试的核心。
    """
    if raw_data_series.empty:
        return pd.Series(dtype="object")

    # 我们在这里“手动”地、一步步地重新计算所有特征
    recalculated_features = pd.Series(
        {
            "均值": raw_data_series.mean(),
            "中位数_Q2": raw_data_series.median(),
            "最大值": raw_data_series.max(),
            "最小值": raw_data_series.min(),
            "Q1": raw_data_series.quantile(0.25),
            "Q3": raw_data_series.quantile(0.75),
            "P10": raw_data_series.quantile(0.10),
            "极差": raw_data_series.max() - raw_data_series.min(),
            # 这里复用了我们的辅助函数，保证了检验标准和生产标准一致
            "超过Q3占时比": calculate_percent_above_q3(raw_data_series),
        }
    )
    return recalculated_features


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
