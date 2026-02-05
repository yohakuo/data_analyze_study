import datetime
import os
import sys

from src import io_v2


def main():
    output_file = "clickhouse_schema.md"

    try:
        client = io_v2.get_clickhouse_client(target="shared")
    except Exception as e:
        print(f"连接失败: {e}")
        sys.exit(1)

    # 获取所有数据库
    all_dbs = client.execute("SHOW DATABASES")
    # 排除系统库，只看业务库
    exclude_dbs = {"system", "information_schema", "default", "INFORMATION_SCHEMA"}
    target_dbs = [db[0] for db in all_dbs if db[0] not in exclude_dbs]

    # 如果你想包含 default，把上面 'default' 去掉即可
    # target_dbs.append('default')

    with open(output_file, "w", encoding="utf-8") as f:
        # 写入文档头部
        f.write("# ClickHouse 数据库结构文档\n\n")
        f.write(
            f"> 导出时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        for db_name in target_dbs:
            print(f"正在处理数据库: {db_name}")
            f.write(f"# 数据库: `{db_name}`\n\n")

            # 获取该库下的所有表
            tables = client.execute(f"SHOW TABLES FROM {db_name}")
            table_list = [t[0] for t in tables]

            if not table_list:
                f.write("*该数据库下暂无表结构*\n\n")
                continue

            for table_name in table_list:
                full_table_name = f"{db_name}.{table_name}"
                print(f"  - 导出表结构: {full_table_name}")

                # 获取建表语句
                try:
                    create_sql = client.execute(f"SHOW CREATE TABLE {full_table_name}")[
                        0
                    ][0]
                except Exception as e:
                    create_sql = f"-- 获取失败: {e}"

                # 写入 Markdown
                f.write(f"## 表: `{table_name}`\n\n")
                f.write("```sql\n")
                f.write(create_sql)
                f.write("\n```\n\n")

                f.write("---\n\n")

    client.disconnect()
    print(f"\n 导出完成！文件已保存为: {os.path.abspath(output_file)}")


if __name__ == "__main__":
    main()
