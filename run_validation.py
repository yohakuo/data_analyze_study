import csv
from zoneinfo import ZoneInfo

import pandas as pd

from src.utils import get_adjacent_feature_rows, get_raw_data_for_hour

# --- 要验证的目标 ---
TABLE_TO_VALIDATE = "humidity_hourly_features_DongNan"
FIELD_TO_VALIDATE = "空气湿度（%）"


def main():
    pd.set_option("display.max_rows", None)  # 确保打印所有特征
    LOCAL_TIMEZONE = ZoneInfo("Asia/Shanghai")

    # 1. 获取脚本计算值 (相邻两行)
    standard_answers = get_adjacent_feature_rows(table_name=TABLE_TO_VALIDATE)

    if standard_answers is not None and len(standard_answers) == 2:
        previous_hour = standard_answers.iloc[0]
        current_hour = standard_answers.iloc[1]

        curr_time_local = (
            pd.to_datetime(current_hour["时间段"]).tz_localize("UTC").tz_convert(LOCAL_TIMEZONE)
        )

        print("✅ 成功获取！将对下面“当前小时”的数据进行验证：")
        print("\n--- 前一小时特征 ---")
        print(previous_hour)
        print("\n--- 当前小时特征 ---")
        print(current_hour)

        # 2. 获取“当前小时”的原始数据用于验证
        timestamp_utc = pd.to_datetime(current_hour["时间段"]).tz_localize("UTC")
        raw_data = get_raw_data_for_hour(
            start_time_utc=timestamp_utc, field_name=FIELD_TO_VALIDATE
        )

        if raw_data:
            # 3. 导出原始数据到 CSV
            filename = f"validation_data_{curr_time_local.strftime('%Y-%m-%d_%H%M')}.csv"
            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([FIELD_TO_VALIDATE])
                for value in raw_data:
                    writer.writerow([value])
            print(f"\n✨ 原始数据已保存到文件: {filename}")

            print("\n" + "=" * 50)
            print("      *** 手动验证极差的时间变化率 ***")
            print("=" * 50)

            prev_range = previous_hour["极差"]
            curr_range = current_hour["极差"]
            print(f"前一小时的极差: {prev_range}")
            print(f"当前小时的极差: {curr_range}")

            manual_rate_of_change = 0.0
            if prev_range != 0:
                manual_rate_of_change = (curr_range - prev_range) / prev_range

            print(
                f"手动计算的变化率: ({curr_range} - {prev_range}) / {prev_range} = {manual_rate_of_change:.4f}"
            )

            script_rate_of_change = current_hour["极差的时间变化率"]
            print(f"脚本计算的变化率: {script_rate_of_change:.4f}")

            if abs(manual_rate_of_change - script_rate_of_change) < 0.0001:
                print("\n✅ 验证成功！“极差的时间变化率”计算正确。")
            else:
                print("\n❌ 验证失败！两个变化率不匹配。")
            print("=" * 50)
        else:
            print("未能从 InfluxDB 获取到这个小时的原始数据。")


if __name__ == "__main__":
    main()
