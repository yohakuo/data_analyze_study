import time
from zoneinfo import ZoneInfo

from src.io import get_clickhouse_client
from src.io_tasks import get_data_for_id, get_distinct_ids, load_features_to_clickhouse
from src.transform import transform_device_data


def run_generic_feature_pipeline(config: dict, id_limit: int = None, specific_ids: list = None):
    """
    按 ID 循环，执行 E-T-L 流程。

    Args:
        config (dict): 包含所有数据库和特征参数的配置字典.
        specific_ids (list, optional):如果不为空，则只计算列表里的 ID，不再去数据库查全量 ID。
        id_limit (int, optional):。
    """

    # 拆分配置字典以便调用
    db_config = config["database"]
    extract_config = config["extract"]
    transform_config = config["transform"]
    load_config = config["load"]

    client = None
    try:
        # 1. [E] 连接
        client = get_clickhouse_client(target=db_config["target"])

        # 2. [E] 获取所有唯一的 ID
        if specific_ids:
            all_device_ids = specific_ids
        else:
            all_device_ids = get_distinct_ids(
                client=client,
                db=extract_config["database"],
                table=extract_config["table"],
                id_column=extract_config["id_column"],
            )

        if not all_device_ids:
            print("未在源表中找到任何 ID，流水线终止。")
            return

        # 3. 【测试】
        if id_limit:
            print(f"【测试模式】，仅处理前 {id_limit} 个 ID。")
            all_device_ids = all_device_ids[:id_limit]

        # start_time = time.time()

        # 4. [Loop] 循环遍历每个 ID
        for i, device_id in enumerate(all_device_ids):
            # print(f"\n--- 正在处理 {i + 1}/{len(all_device_ids)}: (ID: {device_id}) ---")

            # 5. [E] 提取该 ID 的【全部】数据
            raw_df = get_data_for_id(
                client=client,
                db=extract_config["database"],
                table=extract_config["table"],
                device_id=device_id,
                id_column=extract_config["id_column"],
                time_column=extract_config["time_column"],
            )

            if raw_df.empty:
                print(f"   ► (ID: {device_id}) 无数据，跳过。")
                continue

            # 6. [T] 转换数据
            #    (此函数内部调用 src.features.statistica.calculate_features)
            features_df = transform_device_data(
                device_df=raw_df,
                fields_to_process=transform_config["fields_to_process"],
                features_to_calc=transform_config["features_to_calc"],
                freq=transform_config["freq"],
            )

            if features_df.empty:
                print(f"   ► (ID: {device_id}) 未计算出特征，跳过。")
                continue

            # 7. [L] 加载特征
            load_features_to_clickhouse(
                features_df=features_df,
                client=client,
                db=load_config["database"],
                table=load_config["table"],
                stats_cycle=load_config["stats_cycle"],
            )
            print(f"   ► (ID: {device_id}) 处理和存储完毕。")

        # end_time = time.time()
        # print(f"\n🎉 流水线执行完毕！总耗时: {end_time - start_time:.2f} 秒。")

    except Exception as e:
        print(f"\n❌ 流水线发生致命错误: {e}")
    finally:
        if client and client.connection:
            client.disconnect()
            print("\n ClickHouse 连接已关闭。")
