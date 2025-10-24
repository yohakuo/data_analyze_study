"""
存放所有 Excel 导入任务的“定义”。
每个任务包含三个部分：
1. 数据库中的表名 (TABLE_NAME)
2. Excel到数据库的列映射 (MAP)
3. 数据库的表结构 (SCHEMA)
"""

TABLE_1_NAME = "zcp_checkpoint_g1"

TABLE_1_MAP = {
    "采集时间": "time",
    "未检票（人）": "uninspected",
    "已检票（人）": "inspected",
    "数展中心（人）": "digital_exhibition_center",
    "九层楼检票（人）": "nine_storey_building_inspection",
    "小牌坊检票（人）": "small_archway_inspection",
    "在途中（人）": "en_route",
    "数展检票（人）": "digital_exhibition_inspection",
    "总售票（人）": "total_ticket_sold",
}

TABLE_1_SCHEMA = """
CREATE TABLE IF NOT EXISTS {db_name}.`{table_name}`
(
    `time`                          DateTime,
    `uninspected`                   Int32,
    `inspected`                     Int32,
    `digital_exhibition_center`     Int32,
    `nine_storey_building_inspection` Int32,
    `small_archway_inspection`      Int32,
    `en_route`                      Int32,
    `digital_exhibition_inspection` Int32,
    `total_ticket_sold`             Int32,
    `created_at`                    DateTime
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(time)
ORDER BY (time)
"""

TABLE_2_NAME = "mogao_tourist_flow_9_13_to_9_19"

TABLE_2_MAP = {
    "采集时间": "time",
    "实时人数": "real_time_headcount",
    "当日接待": "same_day_reception",
    "近七天接待": "past_seven_days_reception",
    "逗留时长": "dwell_time_range",  # 这个字段是文本，后续需要特殊处理
    "洞窟号": "temple_id",
}

TABLE_2_SCHEMA = """
CREATE TABLE IF NOT EXISTS {db_name}.`{table_name}`
(
    `time`                      DateTime,
    `real_time_headcount`       Int32,
    `same_day_reception`        Int32,
    `past_seven_days_reception` Int32,
    `dwell_time_range`          String, 
    `temple_id`                 String,
    `created_at`                DateTime
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(time)
ORDER BY (temple_id, time)
"""


TABLE_3_NAME = "temple_reception_stats"

TABLE_3_MAP = {
    "采集时间": "time",
    "实时人数": "real_time_headcount",
    "当日接待": "same_day_reception",
    "近七天接待": "past_seven_days_reception",
    "逗留时长": "dwell_time_range",  # 这个字段是文本，后续需要特殊处理
    "洞窟号": "temple_id",
}

TABLE_3_SCHEMA = """
CREATE TABLE IF NOT EXISTS {db_name}.`{table_name}`
(
    `time`                      DateTime,
    `real_time_headcount`       Int32,
    `same_day_reception`        Int32,
    `past_seven_days_reception` Int32,
    `dwell_time_range`          String, 
    `temple_id`                 String,
    `created_at`                DateTime
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(time)
ORDER BY (temple_id, time)
"""
