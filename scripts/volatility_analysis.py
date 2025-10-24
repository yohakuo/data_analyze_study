import datetime
from zoneinfo import ZoneInfo

from src import config
from src.dataset import get_timeseries_data
from src.features import calculate_daily_volatility, find_longest_stable_period
from src.plots import plot_volatility_map


def main():
    """
    执行每日波动性分析的完整流水线
    """
    MEASUREMENT_NAME = "DongNan"
    FIELD_NAME = "空气湿度（%）"
    start_time = datetime.datetime(2021, 1, 1, tzinfo=ZoneInfo("Asia/Shanghai"))
    stop_time = datetime.datetime(2025, 7, 11, tzinfo=ZoneInfo("Asia/Shanghai"))

    # 1. 提取全量原始数据
    raw_df = get_timeseries_data(
        measurement_name=MEASUREMENT_NAME,
        field_name=FIELD_NAME,
        start_time=start_time,
        stop_time=stop_time,
    )

    if raw_df.empty:
        return

    # 2. 计算每日波动性特征
    daily_vol_df = calculate_daily_volatility(raw_df, field_name=config.FIELD_NAME)

    # 3. 分析波动性数据本身，定义阈值
    volatility_stats = daily_vol_df["标准差"].describe()
    stable_threshold = volatility_stats["25%"]

    # 4. 寻找最长的连续平稳期
    start, end, length = find_longest_stable_period(daily_vol_df, stable_threshold)

    # 5. 打印最终的分析报告
    print("\n" + "=" * 50)
    print("      *** 每日波动性分析报告 ***")
    print("=" * 50)
    print("波动性统计摘要 (基于每日标准差):")
    print(volatility_stats.to_string())
    print("-" * 50)
    print(f"平稳阈值 (25%分位数) 被定义为: {stable_threshold:.4f}")
    if start:
        print(f"\n🎉 发现最长的连续平稳期，共 {int(length)} 天！")
        print(f"   建议的分析窗口开始日期: {start}")
        print(f"   建议的分析窗口结束日期: {end}")
    else:
        print("\n未发现连续的平稳期。")
    print("=" * 50)

    # 6. 绘制并保存图表
    plot_volatility_map(daily_vol_df, stable_threshold, field_name=FIELD_NAME)

    # 7. 保存每日波动性数据，方便未来直接使用
    output_filename = f"{config.PROCESSED_DATA_PATH}daily_volatility_map.csv"
    daily_vol_df.to_csv(output_filename)
    print(f"\n✅ 每日波动性数据已保存到文件: {output_filename}")


if __name__ == "__main__":
    main()
