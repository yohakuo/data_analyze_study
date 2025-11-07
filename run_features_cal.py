import datetime
from zoneinfo import ZoneInfo

from src.pipelines.generic_features import run_generic_feature_pipeline

LOCAL_TZ = ZoneInfo("Asia/Shanghai")
SRC_DB = "original_data_processed"
SRC_TABLE = "sensor_temp_humidity_interpolated"
TGT_DB = "features"
TGT_TABLE = "features_hourly_statistical"

PIPELINE_CONFIG = {
    "database": {
        "target": "shared"  # 对应 get_clickhouse_client 的 'shared'
    },
    "extract": {
        "database": SRC_DB,
        "table": SRC_TABLE,
        "id_column": "device_id",  # 循环的 ID
        "time_column": "time",
    },
    "transform": {
        # 你要计算的【字段】
        "fields_to_process": ["humidity", "temperature"],
        # 你要计算的【特征】(必须在 statistica.py 的字典中)
        "features_to_calc": ["均值", "最大值", "最小值", "Q1", "Q3", "极差"],
        "freq": "h",  # 频率
    },
    "load": {
        "database": TGT_DB,
        "table": TGT_TABLE,
        # (批次元数据)
        "temple_id": "ALL",  # 假设这是所有窟的批处理
        "stats_cycle": "hour",
    },
}


if __name__ == "__main__":
    #    设置 id_limit=3，管线将只运行 3 个 ID 后自动停止。
    #    删除 id_limit=3 即可全量运行。
    run_generic_feature_pipeline(PIPELINE_CONFIG, id_limit=3)

    print("--- 特征管线运行结束 ---")
