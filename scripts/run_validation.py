import os

import datacompy
import pandas as pd

# Define file paths at the top for clarity
FILE_MY = "features_dn.csv"
FILE_OTHER = "data_table.csv"
REPORT_FILE = "comparison_report.txt"


def load_and_clean_data(filepath: str, df_name: str, original_tz: str = None):
    """
    加载单个CSV文件，清理数据。
    """
    if not os.path.exists(filepath):
        print(f"❌ Error: File not found for '{df_name}' at '{filepath}'.")
        return None

    print(f"  ► Loading and processing file: {filepath}")

    col_names = [
        "stat_id",
        "temple_id",
        "device_id",
        "stats_start_time",
        "monitored_variable",
        "stats_cycle",
        "feature_key",
        "feature_value",
        "standby_field01",
        "created_at",
    ]

    try:
        df = pd.read_csv(filepath, encoding="utf-8-sig", skiprows=1, names=col_names)

        df_clean = df[["stats_start_time", "feature_value"]].copy()
        df_clean = df_clean.rename(columns={"feature_value": "value"})

        df_clean["stats_start_time"] = pd.to_datetime(df_clean["stats_start_time"])

        # Check if the datetime column is already timezone-aware
        if df_clean["stats_start_time"].dt.tz is None:
            # If it's naive, localize it.
            # Use original_tz if provided, otherwise default to UTC.
            tz = original_tz if original_tz else "UTC"
            df_clean["stats_start_time"] = df_clean["stats_start_time"].dt.tz_localize(tz)

        df_clean["stats_start_time"] = df_clean["stats_start_time"].dt.tz_convert("UTC")

        df_clean["value"] = pd.to_numeric(df_clean["value"], errors="coerce")

        return df_clean.dropna()

    except Exception as e:
        print(f"  ❌ Failed to process file {filepath}: {e}")
        return None


def run_comparison():
    """
    加载、预处理并比较两个清理过的CSV文件。
    """

    df_my_clean = load_and_clean_data(
        filepath=FILE_MY,
        df_name="My Features",
        original_tz="Asia/Shanghai",
    )

    df_other_clean = load_and_clean_data(
        filepath=FILE_OTHER, df_name="Other Features", original_tz="Asia/Shanghai"
    )

    if df_my_clean is None or df_other_clean is None:
        print("\nComparison aborted due to errors in data loading.")
        return

    compare = datacompy.Compare(
        df_my_clean,
        df_other_clean,
        join_columns=["stats_start_time"],
        df1_name="My_Features",
        df2_name="Other_Features",
        abs_tol=0.001,
        rel_tol=0.001,
    )
    # 两个 value 值的差异在千分之一的绝对或相对容差范围内，它们就被认为是匹配的。

    is_match = compare.matches(ignore_extra_columns=True)
    report_content = compare.report()

    try:
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"✔ Full report saved to {REPORT_FILE}")
    except IOError as e:
        print(f"❌ Error saving report to file: {e}")

    if is_match:
        print("✔ Conclusion: DataFrames match according to the specified tolerances.")
    else:
        print("❌ Conclusion: DataFrames do not match.")

    print("--- End of Report ---")


if __name__ == "__main__":
    run_comparison()
