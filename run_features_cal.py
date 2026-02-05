import warnings
from zoneinfo import ZoneInfo

import pandas as pd

from src import io
from src.calculator import FeatureCalculator
from src.transform import (
    process_spectral,
    process_statistical,
    process_volatility,
)

warnings.simplefilter(action="ignore", category=FutureWarning)

# ==============================================================================
#  全局配置 (Configuration)
# ==============================================================================
LOCAL_TZ = ZoneInfo("Asia/Shanghai")
SRC_DB = "original_data"
SRC_TABLE = "sensor_temp_humidity"
TGT_DB = "feature_data"
TGT_TABLE = "test"

# 实例化计算器（无状态工具类）
calc = FeatureCalculator()

PIPELINE_CONFIG = {
    "extract": {
        "database": SRC_DB,
        "table": SRC_TABLE,
        "id_column": "device_id",
        "time_column": "time",
    },
    "load": {
        "database": TGT_DB,
        "table": TGT_TABLE,
    },
    "pipelines": [
        # {
        #     "processor": "high_humidity_exposure",
        #     "fields": ["humidity"],  # 湿度字段
        #     "params": {
        #         "threshold": 62.0,  # 高湿阈值：62%
        #         "freq": "10m",
        #         "cycle_label": "10min",
        #     },
        # },
        # {
        #     "processor": "rainfall_intensity",
        #     "fields": ["rainfall"],  # 降雨量字段
        #     "params": {
        #         "window": "10min",   # 10分钟滑动窗口
        #         "freq": "D",         # 按天统计
        #         "cycle_label": "daily",
        #     },
        # },
        {
            "processor": "statistical",
            "fields": ["humidity", "temperature"],
            "params": {
                "freq": "D",
                "metrics": [
                    "均值",
                    # "最大值",
                    # "最小值",
                    # "极差",
                    # "中位数",
                    # "Q1",
                    # "Q3",
                    # "P10",
                    # "超过Q3占时比",
                ],
            },
        },
        # --- 波动性分析 (按天) ---
        # {
        #     "processor": "volatility",
        #     "fields": ["humidity"],
        #     "params": {
        #         "freq": "d",
        #         "metrics": [
        #             "标准差",
        #             "变异系数",
        #             "一阶自相关",
        #             "平均绝对偏差_均值",
        #             "平均绝对偏差_中位数",
        #         ],
        #         "cycle_label": "d",
        #     },
        # },
        # # --- 周期性/频谱分析 (全量窗口) ---
        # {
        #     "processor": "spectral",
        #     "fields": ["temperature"],
        #     "params": {
        #         # 频谱分析不需要 freq 参数，它分析的是整个切片的规律
        #         "top_n": 3  # 提取前3个主周期
        #     },
        # },
    ],
}

# === 处理器注册表 ===
PROCESSOR_MAP = {
    "statistical": process_statistical,
    "volatility": process_volatility,
    "spectral": process_spectral,
}

TEST_MODE = True


def main(specific_ids=None):
    client = io.get_clickhouse_client(target="shared", database=SRC_DB)

    # 获取设备列表
    if specific_ids:
        device_ids = specific_ids
    else:
        device_ids = io.get_distinct_ids(
            client, SRC_DB, SRC_TABLE, PIPELINE_CONFIG["extract"]["id_column"]
        )
    #  遍历设备
    for device_id in device_ids:
        # [E] 获取原始数据
        raw_df = io.get_data_for_id(
            client,
            SRC_DB,
            SRC_TABLE,
            device_id,
            id_column=PIPELINE_CONFIG["extract"]["id_column"],
            time_column=PIPELINE_CONFIG["extract"]["time_column"],
        )
        if raw_df.empty:
            continue

        # 设置时间索引
        time_col = PIPELINE_CONFIG["extract"]["time_column"]
        if time_col in raw_df.columns:
            raw_df.set_index(time_col, inplace=True)
        else:
            print(f"错误: 数据中缺少时间列 '{time_col}'。")
            continue

        temple_id = (
            raw_df["temple_id"].iloc[0] if "temple_id" in raw_df.columns else "unknown"
        )

        # [T] 特征计算
        features_buffer = []
        for pipe in PIPELINE_CONFIG["pipelines"]:
            proc_name = pipe["processor"]
            fields = pipe["fields"]
            params = pipe["params"]

            handler = PROCESSOR_MAP.get(proc_name)
            if not handler:
                print(f" 警告: 未找到处理器 '{proc_name}'，跳过。")
                continue

            # 对每个字段应用处理器
            for field in fields:
                if field not in raw_df.columns:
                    continue
                try:
                    # === 核心调用 ===
                    df_res = handler(raw_df, field, params)

                    if not df_res.empty:
                        # 补充 io.py 需要的元数据
                        df_res["field_name"] = field
                        features_buffer.append(df_res)

                except Exception as e:
                    print(f"计算出错 : {e}")

        # [Load] 合并并存储
        if features_buffer:
            final_df = pd.concat(features_buffer, ignore_index=True)
            final_df["device_id"] = device_id
            final_df["temple_id"] = temple_id
            if TEST_MODE:
                # 导出到本地 CSV
                freq = PIPELINE_CONFIG["pipelines"][0]["params"].get("freq", "full")
                metrics = PIPELINE_CONFIG["pipelines"][0]["params"].get("metrics", [])
                # 将 metrics (可能为 list) 转为字符串用于文件名，并做简单清洗
                if isinstance(metrics, list):
                    metric_str = "_".join(str(m) for m in metrics) if metrics else "all"
                else:
                    metric_str = str(metrics) if metrics else "all"
                # 只保留字母数字、下划线、短横线、点和中文，其他字符替换为下划线
                import re

                metric_str = re.sub(r"[^\w\u4e00-\u9fff\-_.]", "_", metric_str)
                csv_filename = f"{device_id}_{metric_str}_{freq}.csv"
                final_df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
                print("-" * 50)

            else:
                # 正式写入数据库
                io.load_features_to_clickhouse(
                    features_df=final_df,
                    client=client,
                    db=TGT_DB,
                    table=TGT_TABLE,
                    stats_cycle=None,
                )
        else:
            print(f"   (ID: {device_id}) 无特征生成。")

    print(" 所有任务执行完毕。")


if __name__ == "__main__":
    # 测试设备列表
    test_devices = [
        "20A6",
        "200B",
        "2107",
    ]
    main(specific_ids=test_devices)
