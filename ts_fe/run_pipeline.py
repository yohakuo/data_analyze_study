import src.config as config
from src.dataset import get_humidity_data, store_features_to_clickhouse
from src.features import calculate_hourly_features


def main():
    """
    数据处理流水线主函数
    """
    # 1. 提取数据
    print("--- 正在从 InfluxDB 提取数据 ---")
    raw_df = get_humidity_data()

    if raw_df.empty:
        print("没有提取到数据，流程结束。")
        return

    # 2. 计算特征
    print("---  正在计算小时级特征 ---")
    hourly_features_df = calculate_hourly_features(raw_df, field_name=config.FIELD_NAME)

    # 3. 存储结果
    print("--- 正在将结果存入 ClickHouse ---")
    store_features_to_clickhouse(
        hourly_features_df, table_name=config.HOURLY_FEATURES_TABLE_A
    )


if __name__ == "__main__":
    main()
