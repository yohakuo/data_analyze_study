"""Preview ClickHouse table structure and top rows using clickhouse_connect and project config."""

import sys

from clickhouse_connect import get_client

from src import config

TABLE = "features_caculate"
LIMIT = 5


def main():
    try:
        client = get_client(
            host=config.CLICKHOUSE_HOST,
            port=config.CLICKHOUSE_PORT,
            username=config.CLICKHOUSE_USER,
            password=config.CLICKHOUSE_PASSWORD,
        )

        print(
            f"Connecting to ClickHouse at {config.CLICKHOUSE_HOST}:{config.CLICKHOUSE_PORT}, database: {config.DATABASE_NAME}"
        )

        # Show create table (best-effort)
        try:
            create_stmt = client.command(f"SHOW CREATE TABLE {config.DATABASE_NAME}.`{TABLE}`")
            print("\n=== Table CREATE statement ===\n")
            print(create_stmt)
        except Exception as e:
            print(f"Could not retrieve CREATE TABLE statement: {e}")

        # Show columns via DESCRIBE and print aligned `name : type`
        try:
            desc = client.query_df(f"DESCRIBE TABLE {config.DATABASE_NAME}.`{TABLE}`")
            print("\n=== Table columns ===\n")
            # find max width for name column
            names = desc["name"].astype(str).tolist()
            types = desc["type"].astype(str).tolist()
            max_name_w = max(len(n) for n in names) if names else 0
            for n, t in zip(names, types):
                print(f"{n.ljust(max_name_w)} : {t}")
        except Exception as e:
            print(f"Could not describe table: {e}")

        # Show top rows and print each row as multi-line `field: value` with alignment
        try:
            rows_df = client.query_df(
                f"SELECT * FROM {config.DATABASE_NAME}.`{TABLE}` LIMIT {LIMIT}"
            )
            print("\n=== Top rows ===\n")
            if rows_df.empty:
                print("(no rows)")
            else:
                # compute padding from column names
                col_names = [str(c) for c in rows_df.columns]
                max_col_w = max(len(c) for c in col_names)
                for idx, row in rows_df.head(LIMIT).iterrows():
                    print(f"--- row {idx} ---")
                    for col in col_names:
                        val = row[col]
                        print(f"{col.ljust(max_col_w)} : {val}")
                    print("")
        except Exception as e:
            print(f"Could not query top rows: {e}")

    except Exception as e:
        print(f"Failed to connect to ClickHouse: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
