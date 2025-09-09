import matplotlib
import pandas as pd

matplotlib.use("TkAgg")
import datetime
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import seaborn as sns
from influxdb_client import InfluxDBClient

# --- 数据库连接配置 ---
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "study2025"
INFLUXDB_ORG = "task3"
INFLUXDB_BUCKET = "cave45"
FIELD_NAME = "空气湿度"

CLICKHOUSE_HOST = "localhost"
CLICKHOUSE_PORT = 8123
CLICKHOUSE_USER = "default"
CLICKHOUSE_PASSWORD = "study2025"
DATABASE_NAME = "feature_db"


def get_humidity_data() -> pd.DataFrame:
    with InfluxDBClient(
        url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG
    ) as client:
        query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: 0)
          |> filter(fn: (r) => r["_measurement"] == "adata")
          |> filter(fn: (r) => r["_field"] == "空气湿度")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        print("正在从 InfluxDB 查询所有原始数据，请稍候...")
        df = client.query_api().query_data_frame(query=query, org=INFLUXDB_ORG)

        if not df.empty:
            df_cleaned = df[["_time", "空气湿度"]]
            df_cleaned = df_cleaned.set_index("_time")
            df_cleaned.index = pd.to_datetime(df_cleaned.index)
            df_cleaned["空气湿度"] = pd.to_numeric(
                df_cleaned["空气湿度"], errors="coerce"
            )
            df_cleaned = df_cleaned.dropna()
            return df_cleaned
        return pd.DataFrame()


def analyze_daily_volatility(df: pd.DataFrame):
    """
    计算每日波动性特征，并返回结果 DataFrame.
    """
    if df.empty:
        return pd.DataFrame()

    print("\n正在计算每日波动性特征...")
    daily_stats = df["空气湿度"].resample("D").agg(["std", "mean"])
    daily_stats["变异系数"] = daily_stats["std"] / daily_stats["mean"]
    daily_stats.rename(columns={"std": "每日标准差", "mean": "每日均值"}, inplace=True)
    # 删除可能因为没有数据而产生的空行
    daily_stats.dropna(inplace=True)
    print("每日波动性计算完成！")
    return daily_stats


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    all_data_df = get_humidity_data()
    daily_volatility_df = analyze_daily_volatility(all_data_df)

    if not daily_volatility_df.empty:
        print("\n" + "=" * 50)
        print("      *** 波动性地图自身的统计分析 ***")
        print("=" * 50)
        volatility_stats = daily_volatility_df["每日标准差"].describe()
        print(volatility_stats.to_string())

        stable_threshold = volatility_stats["25%"]
        print(f"\n我们将定义“相对平稳日”的标准为：每日标准差 < {stable_threshold:.4f}")
        print("=" * 50)

        daily_volatility_df["is_stable"] = (
            daily_volatility_df["每日标准差"] < stable_threshold
        )

        # 使用 groupby 和 shift() 来找出所有连续的平稳期
        stable_streaks = (
            daily_volatility_df[daily_volatility_df["is_stable"]]
            .groupby(
                (
                    daily_volatility_df["is_stable"]
                    != daily_volatility_df["is_stable"].shift()
                ).cumsum()
            )
            .size()
        )

        if not stable_streaks.empty:
            longest_streak = stable_streaks.max()
            # 找到最长连胜的 ID
            longest_streak_id = stable_streaks.idxmax()
            # 从原始DataFrame中，根据ID找到对应的日期
            longest_streak_start_date = daily_volatility_df[
                (daily_volatility_df["is_stable"])
                & (
                    (
                        daily_volatility_df["is_stable"]
                        != daily_volatility_df["is_stable"].shift()
                    ).cumsum()
                    == longest_streak_id
                )
            ].index.min()

            longest_streak_end_date = daily_volatility_df[
                (daily_volatility_df["is_stable"])
                & (
                    (
                        daily_volatility_df["is_stable"]
                        != daily_volatility_df["is_stable"].shift()
                    ).cumsum()
                    == longest_streak_id
                )
            ].index.max()

            print(f"\n🎉 发现最长的连续平稳期，共 {int(longest_streak)} 天！")
            print(f"   开始日期: {longest_streak_start_date.date()}")
            print(f"   结束日期: {longest_streak_end_date.date()}")

        plt.figure(figsize=(18, 8))
        plt.plot(
            daily_volatility_df.index,
            daily_volatility_df["每日标准差"],
            marker=".",
            linestyle="-",
            label="每日标准差 (绝对波动)",
        )
        plt.axhline(
            y=stable_threshold,
            color="g",
            linestyle="--",
            label=f"平稳阈值 (25%分位数) = {stable_threshold:.2f}",
        )

        plt.title("每日相对湿度波动性地图", fontproperties="SimHei", fontsize=18)
        plt.xlabel("日期", fontproperties="SimHei", fontsize=12)
        plt.ylabel("每日标准差", fontproperties="SimHei", fontsize=12)
        plt.legend(prop={"family": "SimHei"})
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

        output_filename = "daily_volatility_map.csv"
        daily_volatility_df.to_csv(output_filename)
        print(f"\n✅ 每日波动性数据已保存到文件: {output_filename}")
