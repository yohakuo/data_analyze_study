import os

import matplotlib

matplotlib.use("Agg")  # 使用 'Agg' 引擎，专门用于保存文件，不弹窗
import matplotlib.pyplot as plt

from src import config


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
