# 所有核心的数据计算和转换逻辑,位于dataset之上
import pandas as pd

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
    """
    接收原始数据 DataFrame，为【指定的字段(field_name)】计算小时级统计特征。
    """
    if df.empty or field_name not in df.columns:
        print(f"错误：数据为空或找不到指定的字段 '{field_name}'")
        return pd.DataFrame()

    print(f"\n正在为字段 '{field_name}' 计算小时级统计特征...")

    # 1. 先计算基础特征
    hourly_stats = df.resample("h").agg(
        均值=(field_name, "mean"),
        中位数=(field_name, "median"),
        最大值=(field_name, "max"),
        最小值=(field_name, "min"),
        Q1=(field_name, lambda x: x.quantile(0.25)),
        Q3=(field_name, lambda x: x.quantile(0.75)),
        P10=(field_name, lambda x: x.quantile(0.10)),
    )
    hourly_stats["极差"] = hourly_stats["最大值"] - hourly_stats["最小值"]

    # 2. 计算高级特征一：“超过 Q3 的占时比”
    percent_above_q3 = df[field_name].resample("h").apply(calculate_percent_above_q3)
    hourly_stats["超过Q3占时比"] = percent_above_q3

    # 3. 计算高级特征二：“极差的时间变化率”
    hourly_stats["极差的时间变化率"] = hourly_stats["极差"].pct_change().fillna(0)

    # 4. 最后整理一下列名
    hourly_stats.rename(
        columns={"中位数": "中位数 (Q2)", "P10": "10th百分位数"}, inplace=True
    )

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
        df[df["is_stable"]]
        .groupby((df["is_stable"] != df["is_stable"].shift()).cumsum())
        .size()
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
