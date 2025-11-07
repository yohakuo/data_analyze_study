# 所有核心的数据计算和转换逻辑,位于dataset之上
import numpy as np
import pandas as pd

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
def calculate_features(
    df: pd.DataFrame, field_name: str, feature_list: list, freq: str = "h"
) -> pd.DataFrame:
    """
    接收原始数据 DataFrame，并根据【指定的特征列表】和【时间频率】计算统计特征。
    freq: 'h'-小时, 'D'-天, 'W'-周, 'M'-月
    """
    if df.empty or field_name not in df.columns:
        print(f"错误：数据为空或找不到指定的字段 '{field_name}'")
        return pd.DataFrame()

    print(
        f"\n正在为字段 '{field_name}' 以 '{freq}' 的频率计算指定的 {len(feature_list)} 个特征..."
    )

    agg_functions = {
        feature: FEATURE_CALCULATORS[feature]
        for feature in feature_list
        if feature in FEATURE_CALCULATORS
    }

    if not agg_functions:
        print("错误：指定的特征都不在可计算的列表中。")
        return pd.DataFrame()

    # 使用 .agg() 一次性计算所有被选中的特征
    resampled_stats = df[field_name].resample(freq).agg(**agg_functions)

    #    只有当“最大值”和“最小值”都被点了，我们才能算“极差”
    if "最大值" in resampled_stats.columns and "最小值" in resampled_stats.columns:
        resampled_stats["极差"] = resampled_stats["最大值"] - resampled_stats["最小值"]

    if "极差" in resampled_stats.columns:
        resampled_stats["极差的时间变化率"] = resampled_stats["极差"].pct_change().fillna(0)

    #    只有当“中位数”被点了，重命名它
    if "中位数" in resampled_stats.columns:
        resampled_stats.rename(columns={"中位数": "中位数 (Q2)"}, inplace=True)

    return resampled_stats


def recalculate_single_hour_features(raw_data_series: pd.Series) -> pd.Series:
    """
    接收一个小时的原始数据序列(Series)，重新计算所有（非变化率）特征并返回一个 Series。
    这个函数是自动化测试的核心。
    """
    if raw_data_series.empty:
        return pd.Series(dtype="object")

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
            "超过Q3占时比": calculate_percent_above_q3(raw_data_series),
        }
    )
    return recalculated_features
