# 存放所有可能会变化的配置信息，比如数据库地址、文件名、密码等

# ---  InfluxDB 连接配置 ---
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "study2025"
INFLUXDB_ORG = "task3"
INFLUXDB_BUCKET = "cave45"

# 查询设置
MEASUREMENT_NAME = "DongNan"  # XiBei、DongNan
FIELD_NAME = "空气湿度（%）"  # 空气湿度（%）、空气温度（℃）


# --- 导入CSV文件配置 ---
INFLUX_MEASUREMENT_NAME = "XiBei"
LOCAL_TIMEZONE = "Asia/Shanghai"
RAW_DATA_DIR = "data/raw"
DATA_SUBFOLDER_TO_IMPORT = "xibei"  # 如果要导入子文件夹，指定子文件夹名称，否则留空

# CSV文件的结构定义
TIMESTAMP_COLUMN = "采集时间"
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M:%S"
FIELD_COLUMNS = ["空气温度（℃）", "空气湿度（%）"]
TAG_COLUMNS = []

# 输出相关设置
PROCESSED_DATA_PATH = "data/processed/"
FIGURES_PATH = "figures/"

# --- ClickHouse 连接配置 ---
CLICKHOUSE_HOST = "localhost"
CLICKHOUSE_PORT = 8123
CLICKHOUSE_USER = "default"
CLICKHOUSE_PASSWORD = "study2025"
DATABASE_NAME = "feature_db"
HOURLY_FEATURES_TABLE_A = "humidity_hourly_features_DongNan"
HOURLY_FEATURES_TABLE_B = "humidity_hourly_features_XiBei"
