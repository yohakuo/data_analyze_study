import os

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 指定默认字体为黑体
plt.rcParams["axes.unicode_minus"] = False  # 解决保存图像是负号'-'显示为方块的问题
TARGET_COLUMN = "空气湿度（%）"  # e.g., "空气湿度（%）" or "空气温度（℃）"
DATA_PATH = os.path.join("data", "processed", "preprocessed_data.parquet")
OUTPUT_DIR = os.path.join("reports", "figures")


START_DATE = "2021-01-01"
END_DATE = "2024-01-01"
TRAIN_TEST_SPLIT_DATE = "2023-05-01"


def run_analysis():
    if not os.path.exists(DATA_PATH):
        print("Error")
        return

    df = pd.read_parquet(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        print("Error")
        return

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    series = df.loc[START_DATE:END_DATE, TARGET_COLUMN]
    # 重采样数据到每日
    daily_series = series.resample("D").mean().dropna()
    train_data = daily_series[:TRAIN_TEST_SPLIT_DATE]
    test_data = daily_series[TRAIN_TEST_SPLIT_DATE:]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 画出筛选后的训练集数据图像
    plt.figure(figsize=(14, 7))
    plt.plot(train_data)
    plt.title("空气湿度数据(训练集)")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(True)
    initial_plot_path = os.path.join(OUTPUT_DIR, "initial_training_series.png")
    plt.savefig(initial_plot_path)
    plt.close()

    # 差分
    data_for_acf_pacf = train_data
    d = 1
    data_for_acf_pacf = train_data.diff().dropna()

    # 画出差分后的数据图像
    plt.figure(figsize=(14, 7))
    plt.plot(data_for_acf_pacf)
    plt.title(f"一阶差分后的空气湿度数据 (d={d})")
    plt.xlabel("Date")
    plt.ylabel("Differenced Value")
    plt.grid(True)
    diff_plot_path = os.path.join(OUTPUT_DIR, "differenced_series.png")
    plt.savefig(diff_plot_path)
    plt.close()

    # 绘制 ACF 和 PACF 图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    plot_acf(data_for_acf_pacf, ax=ax1, lags=40)
    ax1.set_title("自相关函数 (ACF)")
    ax1.grid(True)
    plot_pacf(data_for_acf_pacf, ax=ax2, lags=40, method="ywmle")
    ax2.set_title("偏自相关函数 (PACF)")
    ax2.grid(True)
    plt.tight_layout()
    acf_pacf_plot_path = os.path.join(OUTPUT_DIR, "acf_pacf_plots.png")
    plt.savefig(acf_pacf_plot_path)
    plt.close()

    # 模型建立
    p = 4
    q = 4
    model = ARIMA(train_data, order=(p, d, q), freq="D")
    model_fit = model.fit()
    # print(model_fit.summary())

    daily_series = series.resample("D").mean().dropna()
    # 预测的步数等于测试集的长度
    n_forecast = len(test_data)
    forecast_result = model_fit.get_forecast(steps=n_forecast)

    # 获取预测值
    forecast_values = forecast_result.predicted_mean

    # 绘制预测结果图
    plt.figure(figsize=(14, 7))
    # 绘制训练数据
    plt.plot(train_data.index, train_data, label="训练数据")
    # 绘制真实的测试数据
    plt.plot(test_data.index, test_data, label="真实数据 (测试集)", color="orange")
    # 绘制预测数据
    plt.plot(forecast_values.index, forecast_values, label="预测数据", color="green")
    forecast_plot_path = os.path.join(OUTPUT_DIR, "forecast_vs_actual.png")
    plt.savefig(forecast_plot_path)


if __name__ == "__main__":
    # Before running, ensure you have the necessary libraries installed:
    # pip install pandas pyarrow matplotlib statsmodels
    run_analysis()
