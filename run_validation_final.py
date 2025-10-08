import datacompy
import pandas as pd


def run_comparison():
    """
    Loads, preprocesses, and compares two cleaned hour average CSV files.
    """
    try:
        # Define the correct column names, as the header in my_hour_average is broken
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

        # Load my_hour_average.csv by skipping the broken header and assigning names manually
        df_my = pd.read_csv(
            "my_hour_average.csv", encoding="utf-8-sig", skiprows=1, names=col_names
        )

        # Load hour_average.csv normally, as its header is now correct
        df_hour = pd.read_csv("hour_average.csv", encoding="utf-8-sig")

        # --- Preprocessing ---

        # 2. Prepare my_hour_average.csv (the standard)
        df_my_clean = df_my[["stats_start_time", "feature_value"]].copy()
        df_my_clean["stats_start_time"] = pd.to_datetime(
            df_my_clean["stats_start_time"]
        ).dt.tz_localize("UTC")
        df_my_clean.rename(columns={"feature_value": "value"}, inplace=True)

        # 3. Prepare hour_average.csv (to validate)
        df_hour_clean = df_hour[["stats_start_time", "feature_value"]].copy()
        df_hour_clean["stats_start_time"] = (
            pd.to_datetime(df_hour_clean["stats_start_time"])
            .dt.tz_localize("Asia/Shanghai")
            .dt.tz_convert("UTC")
        )
        df_hour_clean.rename(columns={"feature_value": "value"}, inplace=True)

        # Ensure value columns are numeric
        df_my_clean["value"] = pd.to_numeric(df_my_clean["value"], errors="coerce")
        df_hour_clean["value"] = pd.to_numeric(df_hour_clean["value"], errors="coerce")

        df_my_clean.dropna(inplace=True)
        df_hour_clean.dropna(inplace=True)

        # --- Comparison ---

        # 4. Use DataComPy for a detailed comparison
        compare = datacompy.Compare(
            df_my_clean,
            df_hour_clean,
            join_columns=["stats_start_time"],
            df1_name="my_average (Standard)",
            df2_name="hour_average (Cleaned)",
            abs_tol=0.001,
            rel_tol=0.001,
        )
        compare.matches(ignore_extra_columns=True)

        # 5. Print the report
        print("--- Cross-Validation Report ---")
        print(compare.report())

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    run_comparison()
