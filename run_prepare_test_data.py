from datetime import datetime

from src.io import get_ts_clickhouse


def save_data():
    start = datetime(2021, 1, 1, 0, 0, 0)
    stop = datetime(2021, 4, 1, 0, 0, 0)
    db = "original_data"
    table = "sensor_temp_humidity"
    field_name = "humidity"
    device_id = "201A"
    temple_id = "045"

    df = get_ts_clickhouse(
        database_name=db,
        table_name=table,
        field_name=field_name,
        device_id=device_id,
        temple_id=temple_id,
        start_time=start,
        stop_time=stop,
    )

    output_path = "tests/data/real_input.csv"
    df.to_csv(output_path)
    print(f"数据形状: {df.shape}")


if __name__ == "__main__":
    save_data()
