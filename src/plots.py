import os

import matplotlib

matplotlib.use("Agg")  # 使用 'Agg' 引擎，专门用于保存文件，不弹窗
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config


# 波动性分析绘图
def plot_volatility_map(df, threshold, field_name):
    """绘制并保存每日波动性地图"""
    print("\n正在生成“每日波动性地图”图表...")
    plt.figure(figsize=(18, 8))

    plt.plot(
        df.index,
        df["标准差"],
        marker=".",
        linestyle="-",
        label="每日标准差 (绝对波动)",
    )

    # 在图上画出“平稳阈值”线
    plt.axhline(
        y=threshold,
        color="g",
        linestyle="--",
        label=f"平稳阈值 (25%分位数) = {threshold:.2f}",
    )

    plt.title("每日湿度波动性地图 (含平稳阈值)", fontproperties="SimHei", fontsize=18)
    plt.xlabel("日期", fontproperties="SimHei", fontsize=12)
    plt.ylabel("每日标准差", fontproperties="SimHei", fontsize=12)
    plt.legend(prop={"family": "SimHei"})
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # 从 config 中获取路径并保存文件
    filename = f"{config.FIGURES_PATH}daily_volatility_map_{field_name}.png"
    output_dir = os.path.dirname(filename)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(filename)
    print(f"📈 波动性地图已保存为文件: {filename}")


# fft绘图
def plot_fft_spectrum(spectrum_df: pd.DataFrame, top_periods_df: pd.DataFrame, field_name: str):
    """绘制并保存 FFT 频谱图"""
    print("\n正在生成频谱分析图...")

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(15, 7))
    # 过滤掉噪音和超长周期，使图表更清晰
    filtered_df = spectrum_df[(spectrum_df["周期(小时)"] > 2) & (spectrum_df["周期(小时)"] < 2160)]
    plt.plot(filtered_df["周期(小时)"], filtered_df["强度(幅度)"])
    plt.title(f"全量数据频谱分析图 (字段: {field_name})", fontproperties="SimHei", fontsize=16)
    plt.xlabel("周期 (小时)", fontproperties="SimHei", fontsize=12)
    plt.ylabel("幅度", fontproperties="SimHei", fontsize=12)
    plt.grid(True)
    plt.xscale("log")

    # 在图上标记出最强的5个周期
    for index, row in top_periods_df.head(5).iterrows():
        plt.axvline(x=row["周期(小时)"], color="r", linestyle="--", alpha=0.7)
        plt.text(row["周期(小时)"], row["强度(幅度)"], f" {row['周期(小时)']:.1f}h", color="r")

    plot_filename = f"{config.FIGURES_PATH}fft_spectrum_analysis.png"
    plt.savefig(plot_filename)
    print(f"\n📈 频谱分析图已保存为文件: {plot_filename}")
