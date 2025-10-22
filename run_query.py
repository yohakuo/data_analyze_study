import sys

from clickhouse_driver import Client
import pandas as pd

from src import config
from src.dataset import get_clickhouse_client

try:
    DB = config.CLICKHOUSE_SHARED_DB
except AttributeError as e:
    print("❌ 错误：你的 src/config.py 文件中缺少 CLICKHOUSE_SHARED_DB。")
    print(f"({e})")
    sys.exit(1)


print(f"正在尝试连接到数据库: {DB}...")

try:
    # 2. 使用封装的函数来处理连接和回退逻辑
    try:
        # 尝试连接到目标 DB
        client = get_clickhouse_client(DB)

    except Exception:
        # 如果目标 DB 失败 (例如不存在)
        print(f"警告：无法连接到数据库 '{DB}'。")
        print("正在尝试连接到 'default' 数据库...")
        try:
            # 回退到 'default' 数据库
            client = get_clickhouse_client("default")
            DB = "default"  # 更新当前 DB 变量，用于显示提示符
        except Exception as default_e:
            print(f"❌ 连 'default' 数据库也连接失败: {default_e}")
            sys.exit(1)

    print("\n请输入你的 SQL 查询。")
    print("输入 'exit' 或 'quit' 退出。")

    # --- 启动交互式循环 ---
    while True:
        query = input(f"\n{DB}> ")
        if query.lower().strip() in ["exit", "quit"]:
            print("正在退出...")
            break

        # 跳过空行
        if not query.strip():
            continue

        # --- 执行查询 ---
        try:
            # 使用 with_column_types=True
            # 它返回一个元组: (rows, column_info)
            # column_info 是一个包含 (name, type) 元组的列表
            query_result = client.execute(query, with_column_types=True)

            rows = query_result[0]
            columns_info = query_result[1]

            # 检查是否返回了列信息 (SELECT 和 SHOW 语句会返回)
            if columns_info:
                # 从 (name, type) 元组中提取列名
                column_names = [col[0] for col in columns_info]

                if rows:
                    # 正常的 SELECT 或 SHOW 结果
                    df = pd.DataFrame(rows, columns=column_names)
                    print(df.to_string())
                else:
                    # 可能是 SELECT ... LIMIT 0 (有列名但没有数据)
                    df = pd.DataFrame(columns=column_names)
                    print(df.to_string())
                    print("(0 rows)")

            else:
                # 对于没有返回列的语句 (例如 CREATE, INSERT, ALTER)
                # 此时 rows (query_result[0]) 通常是空列表 []
                print("OK.")

        except Exception as e:
            print(f"❌ SQL 错误: {e}")

except Exception as e:
    print(f"❌ 数据库连接失败: {e}")

finally:
    if "client" in locals() and client.connection.connected:
        client.disconnect()
    print("连接已关闭。")
