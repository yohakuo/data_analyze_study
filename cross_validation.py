import csv
from datetime import datetime, timedelta


def parse_utc_time(time_str):
    """Parses UTC time string 'YYYY-MM-DD HH:MM:SS' into a datetime object."""
    try:
        return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def parse_cst_time_and_convert_to_utc(time_str):
    """Parses CST (UTC+8) time string 'YYYY/M/D H:MM' and converts it to a UTC datetime object."""
    try:
        # Handle formats like '2021/1/1 0:00'
        dt_obj = datetime.strptime(time_str, "%Y/%m/%d %H:%M")
        # Subtract 8 hours to convert from UTC+8 to UTC
        return dt_obj - timedelta(hours=8)
    except ValueError:
        return None


def load_hour_average_cst(file_path):
    """Loads the UTC+8 data and converts times to UTC for lookup."""
    data_map = {}
    # print(f"Loading and converting time for {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                cst_time_str = row.get("stats_start_time")
                feature_value_str = row.get("feature_value")

                if not cst_time_str or not feature_value_str:
                    # print(f"Warning: Skipping row {i + 2} in {file_path} due to missing data.")
                    continue

                utc_time = parse_cst_time_and_convert_to_utc(cst_time_str)
                if utc_time is None:
                    # print(
                    #     f"Warning: Could not parse time '{cst_time_str}' in row {i + 2} of {file_path}."
                    # )
                    continue

                try:
                    data_map[utc_time] = float(feature_value_str)
                except (ValueError, TypeError):
                    # print(
                    #     f"Warning: Could not parse feature_value '{feature_value_str}' in row {i + 2} of {file_path}."
                    # )
                    continue
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {e}")
        return None

    # print(f"Loaded {len(data_map)} rows from {file_path}.")
    return data_map


def cross_validate(my_file_path, cst_file_data):
    """
    Cross-validates the UTC data file against the loaded UTC+8 data.
    """
    if cst_file_data is None:
        print("Cannot perform validation because the CST data failed to load.")
        return

    total_rows = 0
    matches = 0
    mismatches = 0
    missing_in_cst_file = 0
    mismatch_samples = []

    # print(f"--- Starting Cross-Validation on {my_file_path} ---")

    try:
        with open(my_file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                total_rows += 1
                utc_time_str = row.get("stats_start_time")
                my_feature_value_str = row.get("feature_value")

                if not utc_time_str or not my_feature_value_str:
                    # print(f"Warning: Skipping row {i + 2} in {my_file_path} due to missing data.")
                    continue

                utc_time = parse_utc_time(utc_time_str)
                if utc_time is None:
                    # print(
                    #     f"Warning: Could not parse time '{utc_time_str}' in row {i + 2} of {my_file_path}."
                    # )
                    continue

                try:
                    my_feature_value = float(my_feature_value_str)
                except (ValueError, TypeError):
                    # print(
                    #     f"Warning: Could not parse feature_value '{my_feature_value_str}' in row {i + 2} of {my_file_path}."
                    # )
                    continue

                if utc_time in cst_file_data:
                    cst_feature_value = cst_file_data[utc_time]

                    # Compare based on rounded values
                    if round(my_feature_value, 4) == round(cst_feature_value, 4):
                        matches += 1
                    else:
                        mismatches += 1
                        if len(mismatch_samples) < 5:  # Store a few samples for debugging
                            mismatch_samples.append(
                                {
                                    "time": utc_time,
                                    "my_value": my_feature_value,
                                    "cst_value": cst_feature_value,
                                }
                            )
                else:
                    missing_in_cst_file += 1

    except FileNotFoundError:
        print(f"Error: File not found at {my_file_path}")
        return
    except Exception as e:
        print(f"An error occurred while reading {my_file_path}: {e}")
        return

    print("\n--- Validation Summary ---")
    print(f"Total rows checked in '{my_file_path}': {total_rows}")
    print(f"Matching values: {matches}")
    print(f"Mismatched values (precision-adjusted): {mismatches}")
    print(f"Rows in '{my_file_path}' not found in 'hour_average.csv': {missing_in_cst_file}")

    if mismatch_samples:
        print("\n--- Mismatch Samples (up to 5) ---")
        for sample in mismatch_samples:
            print(
                f"Time (UTC): {sample['time']}, My Value: {sample['my_value']:.6f}, Their Value: {sample['cst_value']:.6f}"
            )


def check_my_hour_average(file_path):
    """
    Reads and prints the first few rows of specified columns from my_hour_average.csv
    to verify data extraction.
    """
    print(f"--- Checking data extraction from {file_path} ---")
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                 print("Error: Could not read header or file is empty.")
                 return
            print("Successfully opened file. Columns found:", reader.fieldnames)
            print("First 5 rows (full dictionary):")
            for i, row in enumerate(reader):
                if i >= 5:
                    break
                print(f"  Row {i+2}: {row}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # File paths
    my_utc_file = "my_hour_average.csv"
    # cst_file = "hour_average.csv"

    # First, check if we can read the primary file correctly.
    check_my_hour_average(my_utc_file)

    # print("\n--- Running full validation ---")
    # # Run the validation
    # cst_data = load_hour_average_cst(cst_file)
    # cross_validate(my_utc_file, cst_data)
