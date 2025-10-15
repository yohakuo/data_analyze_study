from datetime import datetime
import os

import pandas as pd

from src import config
from src.dataset import get_timeseries_data


def create_preprocessed_dataset(
    measurement_name: str,
    field_name: str,
    start_date_str: str,
    end_date_str: str,
    output_filename: str,
):
    """
    从 InfluxDB 查询数据，执行分钟级补齐、时区转换等预处理，并将结果保存为 CSV 文件。

    Args:
        measurement_name (str): InfluxDB中的 measurement 名称。
        field_name (str): 要查询的字段名称。
        start_date_str (str): 开始日期，格式 'YYYY-MM-DD'。
        end_date_str (str): 结束日期，格式 'YYYY-MM-DD'。
        output_filename (str): 输出的 CSV 文件名。
    """

    # 将日期字符串转换为 datetime 对象
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d").replace(hour=23, minute=59, second=59)

    df = get_timeseries_data(
        measurement_name=measurement_name,
        field_name=field_name,
        start_time=start_dt,
        stop_time=end_dt,
    )

    if df.empty:
        print("❌ 从 InfluxDB 未查询到任何数据，脚本终止。")
        return

    # 分钟级数据补齐
    # 将数据重采样为1分钟频率，这会创建出完整的分钟级时间索引
    # 原本没有数据的时间点，其特征值会变为 NaN
    df_resampled = df.resample("1T").mean()  # 使用 .mean() 来处理同一分钟内有多个数据点的情况
    print(f"数据已重采样至1分钟频率，重采样后行数: {len(df_resampled)}")

    # 规则 1: 使用 ffill() / bfill() 补齐分钟级缺失
    # 先用前一个有效值填充，再用后一个有效值填充，覆盖大部分内部缺失
    df_filled = df_resampled.ffill().bfill()
    print("✅ 规则 1: 已使用 ffill / bfill 完成分钟内插值。 সন")

    # 规则 2: 删除在 ffill/bfill 后仍然完全为空的行 (如果存在)
    # 这种情况通常发生在整个数据集为空，或者某些特定列完全没有数据
    df_dropped = df_filled.dropna(how="all")
    if len(df_filled) != len(df_dropped):
        print(
            f"✅ 规则 2: 已使用 dropna 删除 {len(df_filled) - len(df_dropped)} 个完全为空的周期。 সন"
        )

    # 规则 3: 对初始无法计算的值补零
    # ffill/bfill 后，只有在整个序列开头都无数据的情况下才会存在 NaN
    df_final = df_dropped.fillna(0)
    print("✅ 规则 3: 已使用 fillna(0) 完成初始值补零。 সন")

    # 时区转换
    df_final = df_final.tz_convert("Asia/Shanghai")
    # 创建一个名为 'stats_start_time' 的列来存储转换后的时间字符串
    df_final["stats_start_time"] = df_final.index.strftime("%Y-%m-%d %H:%M:%S")
    print("✅ 时区已转换为 'Asia/Shanghai'，新的时间列 'stats_start_time' 已创建。 সন")
    print("处理后数据预览 (前5行):")
    print(df_final.head())

    # --- 步骤 4: 导出为 CSV 文件 ---
    print("\n--- 步骤 4: 保存处理后的数据 ---")
    os.makedirs(config.PROCESSED_DATA_PATH, exist_ok=True)
    output_path = os.path.join(config.PROCESSED_DATA_PATH, output_filename)

    try:
        # 将 'stats_start_time' 列和特征列导出
        cols_to_export = ["stats_start_time"] + [
            col for col in df_final.columns if col != "stats_start_time"
        ]
        df_final[cols_to_export].to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"✅ 处理后的数据已成功保存到: {output_path}")
    except Exception as e:
        print(f"❌ 保存文件时出错: {e}")


if __name__ == "__main__":
    # --- 配置运行参数 ---
    MEASUREMENT = config.MEASUREMENT_NAME
    FIELD = config.FIELD_NAME
    START_DATE = "2023-01-01"
    END_DATE = "2023-01-31"
    OUTPUT_FILENAME = "data_dongnan.csv"

    create_preprocessed_dataset(
        measurement_name=MEASUREMENT,
        field_name=FIELD,
        start_date_str=START_DATE,
        end_date_str=END_DATE,
        output_filename=OUTPUT_FILENAME,
    )
