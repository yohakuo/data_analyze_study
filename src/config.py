# 存放所有可能会变化的配置信息，比如数据库地址、文件名、密码等

# ================================================================
# ===== 实验室中 ClickHouse 服务器配置 =====
# ================================================================
CLICKHOUSE_SHARED_HOST = "192.168.121.57"
CLICKHOUSE_SHARED_PORT = "5050"
CLICKHOUSE_SHARED_USER = "root"
CLICKHOUSE_SHARED_PASSWORD = "123456"
CLICKHOUSE_SHARED_DB = "original_data"
CLICKHOUSE_MEASUREMENT_NAME = "sensor_temp_humidity"  # sensor_co2

# ====================================================================
# --- 原始数据批量导入配置  ---
# ====================================================================

RAW_FILE_PARSING_CONFIG = {
    # 使用正则表达式从文件名提取信息
    # 组1: device_id (如 20A4, 201A)
    # 组2: temple_id (如 045)
    # 组3: sensor_type_keyword (如 无线二氧化碳传感器 或 无线温湿度传感器)
    "filename_regex": r"^([^_]+)_(\d+)窟_.*(无线温湿度传感器|无线二氧化碳传感器).*\.xlsx?$",
    # Excel 读取配置
    "excel_reading": {
        "header_row": 0,
        # 定义要处理的 Sheet 年份范围 (包含首尾)
        "sheet_year_range": (2020, 2025),
    },
}

# 传感器类型映射与数据表定义
RAW_SENSOR_MAPPING_CONFIG = {
    "无线温湿度传感器": {
        "clickhouse_table": "sensor_temp_humidity",
        # Excel 列名 -> 数据库列名 映射
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
# InfluxDB 连接配置
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "study2025"
INFLUXDB_ORG = "task3"
INFLUXDB_BUCKET = "cave45"

# 导入CSV文件配置
INFLUX_MEASUREMENT_NAME = "DongNan"  # DongNan、XiBei
LOCAL_TIMEZONE = "Asia/Shanghai"

# CSV文件的结构定义
TIMESTAMP_COLUMN = "采集时间"
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M:%S"
FIELD_COLUMNS = ["空气温度（℃）", "空气湿度（%）"]
TAG_COLUMNS = []


# ClickHouse 连接配置
CLICKHOUSE_HOST = "localhost"
CLICKHOUSE_PORT = 8123
CLICKHOUSE_USER = "default"
CLICKHOUSE_PASSWORD = "study2025"
DATABASE_NAME = "feature_db"
HOURLY_FEATURES_TABLE = "humidity_hourly_features_DongNan"  # "humidity_hourly_features_XiBei"

# 输出相关设置
PROCESSED_DATA_PATH = "data/processed/"
FIGURES_PATH = "figures/"

# 查询设置
MEASUREMENT_NAME = "DongNan"  # XiBei、DongNan
FIELD_NAME = "空气湿度（%）"  # 空气湿度（%）、空气温度（℃）、CO₂-校正值（ppm）、CO₂-采集值（ppm）


# ================================================================
# ===== 字段别名/同义词映射表 =====
# ================================================================
# 字典的 key 是数据库中“规范”的、唯一的字段名
# 字典的 value 是一个列表，包含了用户可能使用的所有别名
FIELD_ALIAS_MAP = {
    "空气湿度（%）": ["湿度", "相对湿度", "潮湿程度", "空气湿度"],
    "空气温度": ["温度", "气温", "室温"],
    # 未来可以继续添加其他字段，比如 "二氧化碳浓度": ["二氧化碳", "CO2"]
}
