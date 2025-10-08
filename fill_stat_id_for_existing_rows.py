"""Migration helper: fill stat_id for existing rows where it's missing or zero.

This script prints the ALTER TABLE ... UPDATE statement that you can run on your ClickHouse
server to populate stat_id for existing rows. It does NOT execute destructive changes by default.

Notes:
- Requires clickhouse-connect and proper config in src/config.py
- ALTER TABLE ... UPDATE requires MergeTree tables and certain ClickHouse versions; if your
  ClickHouse version doesn't support UPDATE by expression, use an INSERT ... SELECT INTO tmp
  table approach (script includes suggested fallback).
"""

from clickhouse_connect import get_client

from src import config

TABLE = "features_caculate"
DB = config.DATABASE_NAME

UPDATE_SQL = f"""
ALTER TABLE {DB}.`{TABLE}`
UPDATE stat_id = toUInt64(cityHash64(concat(device_id, toString(stats_start_time), feature_key)))
WHERE stat_id = 0 OR stat_id IS NULL
"""

FALLBACK_SQL = f"""
-- Fallback pattern (safe but requires more disk):
-- 1) CREATE TABLE {DB}.`{TABLE}_tmp` AS {DB}.`{TABLE}` ENGINE = MergeTree() ORDER BY (stats_start_time, device_id, feature_key);
-- 2) INSERT INTO {DB}.`{TABLE}_tmp` SELECT toUInt64(cityHash64(concat(device_id, toString(stats_start_time), feature_key))) AS stat_id, temple_id, device_id, stats_start_time, monitored_variable, stats_cycle, feature_key, feature_value, standby_field01, created_at FROM {DB}.`{TABLE}`;
-- 3) RENAME TABLE {DB}.`{TABLE}` TO {DB}.`{TABLE}_old`, {DB}.`{TABLE}_tmp` TO {DB}.`{TABLE}`;
-- 4) DROP TABLE {DB}.`{TABLE}_old`;
"""


def main():
    print("This script will show SQL to update rows where stat_id is 0 or NULL.")
    print("ClickHouse config from src/config.py will be used:")
    print(f"  host={config.CLICKHOUSE_HOST} port={config.CLICKHOUSE_PORT} db={DB}")
    print("\nSuggested UPDATE statement:\n")
    print(UPDATE_SQL)
    print("\nFallback multi-step SQL (safe) suggestion:\n")
    print(FALLBACK_SQL)

    do_it = input("Do you want me to execute the ALTER TABLE ... UPDATE now? (y/N): ")
    if do_it.lower() != "y":
        print("No changes made. Run the printed SQL on your ClickHouse server when ready.")
        return

    client = get_client(
        host=config.CLICKHOUSE_HOST,
        port=config.CLICKHOUSE_PORT,
        username=config.CLICKHOUSE_USER,
        password=config.CLICKHOUSE_PASSWORD,
    )

    print("Executing UPDATE ... this may take time depending on table size...")
    try:
        client.command(UPDATE_SQL)
        print("UPDATE executed. You may want to run OPTIMIZE TABLE ... FINAL if desired.")
    except Exception as e:
        print(f"Failed to execute UPDATE: {e}")
        print("Consider using the fallback INSERT ... SELECT sequence printed above.")


if __name__ == "__main__":
    main()
