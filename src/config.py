# 存放所有可能会变化的配置信息，比如数据库地址、文件名、密码等

# ================================================================
# ===== 实验室中 ClickHouse 服务器配置 =====
# ================================================================
CLICKHOUSE_SHARED_HOST = "192.168.121.57"
CLICKHOUSE_SHARED_PORT = "5050"
CLICKHOUSE_SHARED_USER = "root"
CLICKHOUSE_SHARED_PASSWORD = "123456"
CLICKHOUSE_SHARED_DB = "original_data"  # original_data_processed

# ====================================================================
# --- 原始数据批量导入配置  ---
# ====================================================================
RAW_FILE_PARSING_CONFIG = {
    # 使用正则表达式从文件名提取信息
    # 组1: device_id (如 20A4, 201A)
    # 组2: temple_id (如 045)
    # 组3: sensor_type_keyword (如 无线二氧化碳传感器 或 无线温湿度传感器)
    "filename_regex": r"^([^_]+)_(\d+)窟_.*(无线温湿度传感器|无线二氧化碳传感器).*\.xlsx?$",
    "excel_reading": {
        "header_row": 0,
        "sheet_year_range": (2020, 2025),
    },
}

# 传感器类型映射与数据表定义
RAW_SENSOR_MAPPING_CONFIG = {
    "无线温湿度传感器": {
        "clickhouse_table": "sensor_temp_humidity",
        "column_mapping": {
            "采集时间": "time",
            "空气温度（℃）": "temperature",
            "空气湿度（%）": "humidity",
        },
    },
    "无线二氧化碳传感器": {
        "clickhouse_table": "sensor_co2",
        "column_mapping": {
            "采集时间": "time",
            "CO₂-校正值（ppm）": "co2_corrected",
            "CO₂-采集值（ppm）": "co2_collected",
        },
    },
}


# ================================================================
# ===== 本地配置 =====
# ================================================================

# ClickHouse 连接配置
CLICKHOUSE_HOST = "localhost"
CLICKHOUSE_PORT = 8123
CLICKHOUSE_USER = "default"
CLICKHOUSE_PASSWORD = "study2025"
