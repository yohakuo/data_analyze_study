import sys

from clickhouse_driver import Client
import pandas as pd

from src import config

try:
    HOST = config.CLICKHOUSE_SHARED_HOST
    PORT = config.CLICKHOUSE_SHARED_PORT
    USER = config.CLICKHOUSE_SHARED_USER
    PASS = config.CLICKHOUSE_SHARED_PASSWORD
    DB = config.CLICKHOUSE_SHARED_DB  # 目标数据库
except AttributeError as e:
    print("❌ 错误：你的 src/config.py 文件中缺少必要的配置。")
    print(f"({e})")
    sys.exit(1)


print(f"正在连接到: {USER}@{HOST}:{PORT}...")

try:
    client = Client(host=HOST, port=PORT, user=USER, password=PASS, database="default")

    # --- 切换到目标数据库 ---
    try:
        client.execute(f"USE `{DB}`")
        print(f"已切换到数据库: {DB}")
    except Exception:
        print(f"警告：无法切换到数据库 '{DB}'。它可能还未创建。")
        DB = "default"

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
            result = client.execute(query)

            # 检查查询是否返回了数据 (例如 SELECT)
            if result and client.last_query.column_names:
                df = pd.DataFrame(result, columns=client.last_query.column_names)
                print(df.to_string())  # .to_string() 确保所有行列都显示
            else:
                # 对于非 SELECT (CREATE, INSERT, ALTER, SHOW)
                if result:
                    print(result)  # SHOW DATABASES 会返回元组列表
                print("OK.")

        except Exception as e:
            print(f"❌ SQL 错误: {e}")

except Exception as e:
    print(f"❌ 数据库连接失败: {e}")

finally:
    if "client" in locals() and client.connection.connected:
        client.disconnect()
    print("连接已关闭。")
