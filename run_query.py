import re  # 导入 re 模块，用于更灵活地解析
import sys

from clickhouse_driver import Client
import pandas as pd

from src import config
from src.io import get_clickhouse_client

try:
    DB = config.CLICKHOUSE_SHARED_DB
except AttributeError as e:
    print("❌ 错误：你的 src/config.py 文件中缺少 CLICKHOUSE_SHARED_DB。")
    print(f"({e})")
    sys.exit(1)


try:
    try:
        client = get_clickhouse_client(target="shared")

    except Exception:
        try:
            client = get_clickhouse_client(target="default")
            DB = "default"  # 更新当前 DB 变量，用于显示提示符
        except Exception as default_e:
            print(f"❌ 连 'default' 数据库也连接失败: {default_e}")
            sys.exit(1)

    print("\n请输入 SQL 查询。")
    print("输入 'exit' 或 'quit' 退出。")

    # --- 启动交互式循环 ---
    while True:
        query = input(f"\n{DB}> ")
        if query.lower().strip() in ["exit", "quit"]:
            print("正在退出...")
            break

        # 跳过空行
        query_stripped = query.strip()
        if not query_stripped:
            continue

        # 匹配 'USE database_name' (忽略大小写, 处理分号)

        temp_new_db = None  # 暂存新的数据库名称

        # 使用正则表达式来匹配 'USE' 命令
        # re.IGNORECASE (i) 忽略大小写
        # re.DOTALL (s) 让 . 也能匹配换行符
        # ^\s*USE\s+     -> 以 'USE' 开头 (允许前后有空格)
        # ([a-zA-Z0-9_]+) -> 捕获数据库名称 (只允许标准字符)
        match = re.match(
            r"^\s*USE\s+([a-zA-Z0-9_]+)\s*;?\s*$", query_stripped, re.IGNORECASE | re.DOTALL
        )

        if match:
            # 如果匹配成功，match.group(1) 就是捕获到的数据库名
            temp_new_db = match.group(1)

        # --- 执行查询 ---
        try:
            # 使用 with_column_types=True
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

                # --- [!!! 修改逻辑开始 !!!] ---
                # 如果这是一个 'USE' 命令，并且它成功执行了 (没抛异常)
                if temp_new_db:
                    # 正式更新 DB 变量
                    DB = temp_new_db
                    print("OK.")
                else:
                    # 否则，就是其他普通的 OK 命令
                    print("OK.")
                # --- [!!! 修改逻辑结束 !!!] ---

        except Exception as e:
            # 如果 'USE' 命令失败 (例如数据库不存在)，
            # 异常会在这里被捕获，DB 变量不会被更新。
            print(f"❌ SQL 错误: {e}")

except Exception as e:
    print(f"❌ 数据库连接失败: {e}")

finally:
    if "client" in locals() and client.connection.connected:
        client.disconnect()
    print("连接已关闭。")
