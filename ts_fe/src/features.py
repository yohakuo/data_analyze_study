# 所有核心的数据计算和转换逻辑都放在这里
import pandas as pd


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
