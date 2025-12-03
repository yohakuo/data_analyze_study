import datetime
import warnings
from zoneinfo import ZoneInfo

from src.pipelines.generic_features import run_generic_feature_pipeline

warnings.simplefilter(action="ignore", category=FutureWarning)

LOCAL_TZ = ZoneInfo("Asia/Shanghai")
SRC_DB = "original_data"
SRC_TABLE = "sensor_temp_humidity"
TGT_DB = "feature_data"
TGT_TABLE = "sensor_feature_data"
# TGT_Temple = "045"
STATS_CYCLE = "hour"

PIPELINE_CONFIG = {
    "database": {"target": "shared"},
    "extract": {
        "database": SRC_DB,
        "table": SRC_TABLE,
        "id_column": "device_id",  # 循环的 ID
        "time_column": "time",
    },
    "transform": {
        "fields_to_process": ["humidity", "temperature"],
        # 计算的【特征】(必须在 statistica.py 的字典中)
        "features_to_calc": [
            "中位数",
            "最大值",
            "最小值",
            "标准差",
        ],  # "均值","中位数", "最大值", "最小值", "标准差"，"Q1", "Q3","极差"
        "freq": "h",  # 频率
    },
    "load": {
        "database": TGT_DB,
        "table": TGT_TABLE,
        "stats_cycle": STATS_CYCLE,
    },
}

device_list = ["201A", "2022", "2178"]  #

if __name__ == "__main__":
    run_generic_feature_pipeline(PIPELINE_CONFIG, specific_ids=device_list)

    print("--- 运行结束 ---")
