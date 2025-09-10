# 计算波动性特征找到的时间窗口
import datetime
from zoneinfo import ZoneInfo

from src import config
from src.dataset import get_timeseries_data
from src.features import analyze_with_fft
from src.plots import plot_fft_spectrum

ANALYSIS_START_TIME_LOCAL = datetime.datetime(2022, 1, 4, tzinfo=ZoneInfo("Asia/Shanghai"))
ANALYSIS_STOP_TIME_LOCAL = datetime.datetime(2022, 1, 14, tzinfo=ZoneInfo("Asia/Shanghai"))


def main():
    """
    执行对指定时间窗口的数据进行 FFT 周期性分析的流水线。
    """
    # 1. 提取指定窗口内的原始数据
    print("--- 正在提取指定窗口内的数据 ---")
    raw_df = get_timeseries_data(
        measurement_name=config.MEASUREMENT_NAME,
        field_name=config.FIELD_NAME,
        start_time=ANALYSIS_START_TIME_LOCAL,
        stop_time=ANALYSIS_STOP_TIME_LOCAL,
    )

    if raw_df.empty:
        print("指定时间段内没有数据，分析终止。")
        return

    # 2. 进行 FFT 分析
    spectrum_df = analyze_with_fft(raw_df, field_name=config.FIELD_NAME)

    # 3. 找出最强的周期并打印报告
    top_10_periods = spectrum_df.sort_values(by="强度(幅度)", ascending=False).head(10)
    print("\n" + "=" * 40)
    print("      *** 周期性分析报告 ***")
    print("=" * 40)
    print("在分析窗口内，信号最强的 Top 10 周期为：")
    print(top_10_periods.to_string(index=False))
    print("=" * 40)

    # 4. 绘制并保存频谱图
    print("--- 正在生成并保存图表 ---")
    plot_fft_spectrum(spectrum_df, top_10_periods, field_name=config.FIELD_NAME)


if __name__ == "__main__":
    main()
