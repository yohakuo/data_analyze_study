# run_import_excel_data.py
import pandas as pd

import src.config as config
from src.dataset import (
    create_table_if_not_exists,
    get_clickhouse_client,
    load_to_clickhouse,
    read_excel_data,
)
from src.table_definitions import (
    TABLE_1_MAP,
    TABLE_1_NAME,
    TABLE_1_SCHEMA,
    TABLE_2_MAP,
    TABLE_2_NAME,
    TABLE_2_SCHEMA,
    # (未来有新表, 在这里导入 TABLE_3_... )
)
from src.utils import process_dataframe

# --- 配置文件路径 ---
FILE_1_PATH = "D:/dataset/游客量数据/zcp_小牌坊检票口_莫高窟闸机1.xlsx"
FILE_2_PATH = "D:/dataset/游客量数据/莫高窟洞窟游客数据9-13~9-19.xlsx"
# FILE_3_PATH = "data/..."


def process_task_1(client):
    """
    zcp_小牌坊...
    支持读取所有 sheet 并合并。
    增加了 float -> int 的类型转换。
    """
    try:
        db_name = config.CLICKHOUSE_SHARED_DB
        # 1. 建表
        if not create_table_if_not_exists(client, db_name, TABLE_1_NAME, TABLE_1_SCHEMA):
            print("失败: 无法创建表。")
            return

        # 2. 读取 (E - Extract)
        all_sheets_dict = pd.read_excel(FILE_1_PATH, sheet_name=None, header=0)

        # 3. 转换 (T - Transform)
        list_of_processed_dfs = []

        for sheet_name, df_raw in all_sheets_dict.items():
            print(f"  > G正在处理 sheet: '{sheet_name}' (读取 {len(df_raw)} 行)")

            if df_raw.empty:
                print(f"  > Sheet '{sheet_name}' 为空，跳过。")
                continue

            # (处理采集时间列的 '#####' 问题)
            if "采集时间" in df_raw.columns:
                df_raw["采集时间"] = pd.to_datetime(df_raw["采集时间"], errors="coerce")
                df_raw.dropna(subset=["采集时间"], inplace=True)  # 删除无法解析时间的行

            df_processed = process_dataframe(df_raw, TABLE_1_MAP)

            int_columns = [
                "uninspected",
                "inspected",
                "digital_exhibition_center",
                "nine_storey_building_inspection",
                "small_archway_inspection",
                "en_route",
                "digital_exhibition_inspection",
                "total_ticket_sold",
            ]

            for col in int_columns:
                if col in df_processed.columns:
                    # .fillna(0) 处理可能存在的空值 (NaN)
                    # .astype(int) 将浮点数 (如 6438.0) 转换为整数 (6438)
                    df_processed[col] = df_processed[col].fillna(0).astype(int)

            list_of_processed_dfs.append(df_processed)

        # 4. 合并所有处理过的 DataFrame
        if not list_of_processed_dfs:
            print("[任务 1] 警告: 没有在任何 sheet 中找到可加载的数据。")
            return

        final_df = pd.concat(list_of_processed_dfs, ignore_index=True)

        print(f"所有 sheets 合并完毕，总共 {len(final_df)} 条记录将被加载。")

        # 5. 加载 (L - Load)
        load_to_clickhouse(client, final_df, TABLE_1_NAME)

    except Exception as e:
        print(f"--- [任务 1] 发生意外错误: {e} ---")


def process_task_2(client):
    try:
        db_name = config.CLICKHOUSE_SHARED_DB
        if not create_table_if_not_exists(client, db_name, TABLE_2_NAME, TABLE_2_SCHEMA):
            print("失败: 无法创建表。")
            return

        df_raw = read_excel_data(FILE_2_PATH, header_row=0, dtypes={"洞窟号": str})
        df_processed = process_dataframe(df_raw, TABLE_2_MAP)
        load_to_clickhouse(client, df_processed, TABLE_2_NAME)

    except Exception as e:
        print(f"--- 发生意外错误: {e} ---")


def main():
    try:
        DB = config.CLICKHOUSE_SHARED_DB
        client = get_clickhouse_client(DB)
        process_task_1(client)
        # process_task_2(client)

    except Exception as e:
        print(f"\n[主程序] 发生未捕获的严重错误: {e}")
        print("脚本已终止。")
    finally:
        if client:
            client.disconnect()


if __name__ == "__main__":
    main()
