#!/usr/bin/env python3
"""
ClickHouse 表管理脚本
用于查询所有表并删除空表
"""

import clickhouse_connect

from src.config import (
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USER,
    DATABASE_NAME,
)


def connect_to_clickhouse():
    """连接到 ClickHouse 数据库"""
    try:
        client = clickhouse_connect.get_client(
            host=CLICKHOUSE_HOST,
            port=CLICKHOUSE_PORT,
            username=CLICKHOUSE_USER,
            password=CLICKHOUSE_PASSWORD,
            database=DATABASE_NAME,
        )
        print(f"✅ 成功连接到 ClickHouse 数据库: {DATABASE_NAME}")
        return client
    except Exception as e:
        print(f"❌ 连接 ClickHouse 失败: {e}")
        return None


def get_all_tables(client):
    """获取数据库中的所有表"""
    try:
        # 查询系统表获取所有表名
        query = f"""
        SELECT name 
        FROM system.tables 
        WHERE database = '{DATABASE_NAME}'
        """
        result = client.query(query)
        tables = [row[0] for row in result.result_rows]
        print(f"📊 发现 {len(tables)} 张表: {tables}")
        return tables
    except Exception as e:
        print(f"❌ 查询表列表失败: {e}")
        return []


def get_table_row_count(client, table_name):
    """获取表的行数"""
    try:
        query = f"SELECT count() FROM {table_name}"
        result = client.query(query)
        row_count = result.result_rows[0][0]
        print(f"  表 {table_name} 有 {row_count} 行数据")
        return row_count
    except Exception as e:
        print(f"❌ 查询表 {table_name} 行数失败: {e}")
        return -1


def delete_table(client, table_name):
    """删除表"""
    try:
        query = f"DROP TABLE IF EXISTS {table_name}"
        client.command(query)
        print(f"🗑️  已删除空表: {table_name}")
        return True
    except Exception as e:
        print(f"❌ 删除表 {table_name} 失败: {e}")
        return False


def main():
    # 连接到 ClickHouse
    client = connect_to_clickhouse()
    if not client:
        return

    # 获取所有表
    tables = get_all_tables(client)
    if not tables:
        print("⚠️  数据库中没有表")
        return

    empty_tables = []

    # 检查每个表的行数
    print("\n📋 检查各表数据量:")
    for table in tables:
        row_count = get_table_row_count(client, table)
        if row_count == 0:
            empty_tables.append(table)

    # 处理空表
    if empty_tables:
        print(f"\n⚠️  发现 {len(empty_tables)} 张空表: {empty_tables}")
        confirm = input("是否要删除这些空表？(y/N): ")

        if confirm.lower() == "y":
            for table in empty_tables:
                delete_table(client, table)
            print("✅ 空表删除完成")
        else:
            print("❌ 取消删除操作")
    else:
        print("✅ 没有发现空表")

    client.close()


if __name__ == "__main__":
    main()
