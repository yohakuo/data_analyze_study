# 波动性特征
import pandas as pd


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
