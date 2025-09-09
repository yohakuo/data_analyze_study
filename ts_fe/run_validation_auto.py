# 独立，直接结果
import csv
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from src.validation import get_adjacent_feature_rows, get_raw_data_for_hour

# --- 要验证的目标 ---
TABLE_TO_VALIDATE = "humidity_hourly_features_DongNan"
FIELD_TO_VALIDATE = "空气湿度（%）"


def calculate_percent_above_q3(series):
    """接收一个小时的原始数据序列(series)，计算超过Q3的数据点占比。"""
    if len(series) < 4:
        return 0.0
    q3 = series.quantile(0.75)
    if q3 == series.max():
        return 0.0
    count_above_q3 = (series > q3).sum()
    percent_above_q3 = count_above_q3 / len(series)
    return percent_above_q3


def recalculate_single_hour_features(
    raw_data_series: pd.Series, field_name: str
) -> pd.Series:
    """接收一个小时的原始数据序列，重新计算所有特征并返回一个 Series"""
    if raw_data_series.empty:
        return pd.Series(dtype="object")

    # 计算基础统计特征
    mean_val = raw_data_series.mean()
    median_val = raw_data_series.median()
    max_val = raw_data_series.max()
    min_val = raw_data_series.min()
    q1_val = raw_data_series.quantile(0.25)
    q3_val = raw_data_series.quantile(0.75)
    p10_val = raw_data_series.quantile(0.10)

    # 计算衍生特征
    range_val = max_val - min_val
    percent_above_q3_val = calculate_percent_above_q3(raw_data_series)

    # 将结果打包成一个 Series
    recalculated_features = pd.Series(
        {
            "均值": mean_val,
            "中位数_Q2": median_val,
            "最大值": max_val,
            "最小值": min_val,
            "Q1": q1_val,
            "Q3": q3_val,
            "P10": p10_val,
            "极差": range_val,
            "超过Q3占时比": percent_above_q3_val,
        }
    )
    return recalculated_features


def main():
    pd.set_option("display.max_rows", None)
    LOCAL_TIMEZONE = ZoneInfo("Asia/Shanghai")

    standard_answers = get_adjacent_feature_rows(table_name=TABLE_TO_VALIDATE)

    if standard_answers is not None and len(standard_answers) == 2:
        previous_hour = standard_answers.iloc[0]
        current_hour = standard_answers.iloc[1]

        curr_time_local = (
            pd.to_datetime(current_hour["时间段"])
            .tz_localize("UTC")
            .tz_convert(LOCAL_TIMEZONE)
        )

        print(
            f"\n--- 正在验证时间段为 {curr_time_local.strftime('%Y-%m-%d %H:%M:%S')} 的数据 ---"
        )

        timestamp_utc = pd.to_datetime(current_hour["时间段"]).tz_localize("UTC")
        raw_data_list = get_raw_data_for_hour(
            start_time_utc=timestamp_utc, field_name=FIELD_TO_VALIDATE
        )

        if not raw_data_list:
            print("❌ 错误：未能从 InfluxDB 获取到这个小时的原始数据，无法进行验证。")
            return

        # 将原始数据列表转换为 Series，以便我们进行计算
        raw_data_series = pd.Series(raw_data_list)

        # 1. 重新计算当前小时的所有基础和高级特征
        recalculated_features = recalculate_single_hour_features(
            raw_data_series, FIELD_TO_VALIDATE
        )

        # 2. 对比两个结果，并生成报告
        print("\n" + "=" * 60)
        print("         *** 自动化特征验证报告 ***")
        print("=" * 60)
        total_checks = 0
        failed_checks = 0

        # 逐个特征进行对比
        for feature_name, recalculated_value in recalculated_features.items():
            stored_value = current_hour[feature_name]
            # 使用 numpy 的 isclose 函数进行浮点数比较，更可靠
            if np.isclose(recalculated_value, stored_value, atol=1e-5, rtol=1e-5):
                print(
                    f"✅ {feature_name:<16}: 计算值={recalculated_value:8.4f}  |  存储值={stored_value:8.4f}  |  结果: 正确"
                )
                total_checks += 1
            else:
                print(
                    f"❌ {feature_name:<16}: 计算值={recalculated_value:8.4f}  |  存储值={stored_value:8.4f}  |  结果: 错误"
                )
                total_checks += 1
                failed_checks += 1

        # 3. 验证最复杂的“极差的时间变化率”
        prev_range = previous_hour["极差"]
        curr_range = recalculated_features["极差"]

        manual_rate_of_change = 0.0
        if prev_range != 0:
            manual_rate_of_change = (curr_range - prev_range) / prev_range

        script_rate_of_change = current_hour["极差的时间变化率"]

        print("\n" + "-" * 60)
        if np.isclose(manual_rate_of_change, script_rate_of_change, atol=1e-5):
            print(
                f"✅ {'极差的时间变化率':<22}: 计算值={manual_rate_of_change:8.4f}  |  存储值={script_rate_of_change:8.4f}  |  结果: 正确"
            )
            total_checks += 1
        else:
            print(
                f"❌ {'极差的时间变化率':<22}: 计算值={manual_rate_of_change:8.4f}  |  存储值={script_rate_of_change:8.4f}  |  结果: 错误"
            )
            total_checks += 1
            failed_checks += 1

        print("=" * 60)
        if failed_checks == 0:
            print(f"🎉 恭喜！所有 {total_checks} 项特征验证全部通过！")
        else:
            print(f"⚠️ 警告！{failed_checks} 项特征验证失败，请检查。")
        print("=" * 60)

        # 为了方便你查看，我们还是生成原始数据的csv
        filename = f"validation_data_{curr_time_local.strftime('%Y-%m-%d_%H%M')}.csv"
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([FIELD_TO_VALIDATE])
            for value in raw_data_list:
                writer.writerow([value])
        print(f"\n✨ 原始数据已保存到文件: {filename}，便于手动复查。")


if __name__ == "__main__":
    main()
