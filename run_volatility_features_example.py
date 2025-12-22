"""
Volatility Features Calculation Example

This script demonstrates how to use the calculate_volatility_features function
from the FeatureCalculator class. It shows:

1. Loading time series data (from CSV or ClickHouse)
2. Calculating daily volatility features
3. Analyzing the results
4. Finding stable periods
5. Using the new statistical functions (MAD, CV, autocorrelation)

Requirements:
    - pandas
    - numpy
    - src.features.calculator

Usage:
    python run_volatility_features_example.py
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.features.calculator import FeatureCalculator

try:
    from src.io import get_ts_clickhouse
    from src import config

    CLICKHOUSE_AVAILABLE = True
except ImportError:
    CLICKHOUSE_AVAILABLE = False
    print("Note: ClickHouse utilities not available. Will use CSV/parquet data only.")


def load_data_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Load time series data from a CSV file.

    Args:
        csv_path: Path to the CSV file. Must have a 'time' column
                 that will be used as the index.

    Returns:
        DataFrame with DatetimeIndex
    """
    print(f"\n{'=' * 70}")
    print("LOADING DATA FROM CSV")
    print(f"{'=' * 70}")

    try:
        df = pd.read_csv(csv_path, parse_dates=["time"], index_col="time")
        # Remove timezone if present for consistency
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.sort_index()
        print(f"[OK] Loaded {len(df)} data points from {csv_path}")
        print(f"  Time range: {df.index[0]} to {df.index[-1]}")
        print(f"  Columns: {list(df.columns)}")
        return df

    except FileNotFoundError:
        print(f"[ERROR] File '{csv_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Error loading CSV: {e}")
        sys.exit(1)


def load_data_from_clickhouse(
    database_name: str,
    table_name: str,
    field_name: str,
    device_id: str = None,
    temple_id: str = None,
    start_time: datetime = None,
    stop_time: datetime = None,
) -> pd.DataFrame:
    """
    Load time series data from ClickHouse.

    Args:
        database_name: ClickHouse database name
        table_name: ClickHouse table name
        field_name: Field name to extract
        device_id: Optional device ID filter
        temple_id: Optional temple ID filter
        start_time: Optional start time filter
        stop_time: Optional stop time filter

    Returns:
        DataFrame with DatetimeIndex
    """
    if not CLICKHOUSE_AVAILABLE:
        print("[ERROR] ClickHouse utilities not available.")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print("LOADING DATA FROM CLICKHOUSE")
    print(f"{'=' * 70}")

    try:
        df = get_ts_clickhouse(
            database_name=database_name,
            table_name=table_name,
            field_name=field_name,
            device_id=device_id,
            temple_id=temple_id,
            start_time=start_time,
            stop_time=stop_time,
        )

        if df.empty:
            print("[ERROR] No data returned from ClickHouse.")
            sys.exit(1)

        # Remove timezone if present
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.sort_index()
        print(f"[OK] Loaded {len(df)} data points from ClickHouse")
        print(f"  Database: {database_name}, Table: {table_name}")
        print(f"  Time range: {df.index[0]} to {df.index[-1]}")
        print(f"  Field: {field_name}")
        return df

    except Exception as e:
        print(f"[ERROR] Error loading from ClickHouse: {e}")
        sys.exit(1)


def generate_sample_data(days: int = 30, freq: str = "h") -> pd.DataFrame:
    """
    Generate synthetic time series data for demonstration.

    Args:
        days: Number of days of data to generate
        freq: Frequency ('h' for hourly, '30min' for 30 minutes)

    Returns:
        DataFrame with DatetimeIndex and sample data
    """
    print(f"\n{'=' * 70}")
    print("GENERATING SAMPLE DATA")
    print(f"{'=' * 70}")

    import numpy as np

    start_date = datetime(2024, 1, 1)
    periods = days * 24 if freq == "h" else days * 48
    date_range = pd.date_range(start=start_date, periods=periods, freq=freq)

    # Generate synthetic data with daily cycle and noise
    hours = np.arange(len(date_range))
    daily_cycle = 5 * np.sin(2 * np.pi * hours / 24)
    trend = 0.01 * hours
    noise = np.random.normal(0, 1.5, len(date_range))

    temperature = 20 + daily_cycle + trend + noise

    df = pd.DataFrame({"temperature": temperature}, index=date_range)
    print(f"[OK] Generated {len(df)} data points")
    print(f"  Time range: {df.index[0]} to {df.index[-1]}")
    print(f"  Temperature range: {df['temperature'].min():.2f}C to {df['temperature'].max():.2f}C")

    return df


def example_volatility_features():
    """
    Main example: Calculate and analyze volatility features.
    """
    # ====================================================================
    # Step 1: Load or generate data
    # ====================================================================

    # Option 1: Use sample data (for demonstration)
    df = generate_sample_data(days=60, freq="h")
    field_name = "temperature"

    # Option 2: Load from CSV (uncomment to use)
    # csv_path = "tests/data/real_input.csv"
    # df = load_data_from_csv(csv_path)
    # field_name = "humidity"  # or whatever column name you have

    # Option 3: Load from ClickHouse (uncomment to use)
    # df = load_data_from_clickhouse(
    #     database_name="original_data",
    #     table_name="sensor_temp_humidity",
    #     field_name="humidity",
    #     device_id="201A",
    #     temple_id="045",
    #     start_time=datetime(2021, 1, 1),
    #     stop_time=datetime(2021, 4, 1),
    # )
    # field_name = "humidity"

    # ====================================================================
    # Step 2: Calculate volatility features
    # ====================================================================

    calculator = FeatureCalculator()

    # Calculate daily volatility features
    volatility = calculator.calculate_volatility_features(df, field_name=field_name)

    if volatility.empty:
        print("[ERROR] No volatility features calculated. Check your data.")
        return

    print(f"[OK] Calculated volatility features for {len(volatility)} days")
    print(f"\nAvailable metrics:")
    for col in volatility.columns:
        print(f"  - {col}")

    # ====================================================================
    # Step 3: Display results
    # ====================================================================
    print("\n--- Step 3: Volatility Features Overview ---")
    print("\nFirst 10 days:")
    print(volatility.head(10).to_string())

    print("\n--- Summary Statistics ---")
    print(f"Average daily mean: {volatility['均值'].mean():.4f}")
    print(f"Average daily std: {volatility['标准差'].mean():.4f}")
    print(f"Average MAD (from mean): {volatility['平均绝对偏差_均值'].mean():.4f}")
    print(f"Average MAD (from median): {volatility['平均绝对偏差_中位数'].mean():.4f}")
    print(f"Average coefficient of variation: {volatility['变异系数'].mean():.4f}")
    print(f"Average first-order autocorrelation: {volatility['一阶自相关'].mean():.4f}")

    # ====================================================================
    # Step 4: Analyze volatility patterns
    # ====================================================================
    print("\n--- Step 4: Volatility Pattern Analysis ---")

    # Find most and least volatile days
    most_volatile_idx = volatility["标准差"].idxmax()
    least_volatile_idx = volatility["标准差"].idxmin()

    print(f"\nMost volatile day: {most_volatile_idx.date()}")
    print(f"  Standard deviation: {volatility.loc[most_volatile_idx, '标准差']:.4f}")
    print(f"  Mean: {volatility.loc[most_volatile_idx, '均值']:.4f}")
    print(f"  Coefficient of variation: {volatility.loc[most_volatile_idx, '变异系数']:.4f}")

    print(f"\nLeast volatile day: {least_volatile_idx.date()}")
    print(f"  Standard deviation: {volatility.loc[least_volatile_idx, '标准差']:.4f}")
    print(f"  Mean: {volatility.loc[least_volatile_idx, '均值']:.4f}")
    print(f"  Coefficient of variation: {volatility.loc[least_volatile_idx, '变异系数']:.4f}")

    # ====================================================================
    # Step 5: Find stable periods
    # ====================================================================
    print("\n--- Step 5: Finding Stable Periods ---")

    # Use 30th percentile as threshold for stability
    threshold = volatility["标准差"].quantile(0.3)
    print(f"Stability threshold (30th percentile): {threshold:.4f}")

    start_date, end_date, length = calculator.find_stable_period(volatility, threshold)

    if start_date is not None:
        print(f"\n[OK] Longest stable period found:")
        print(f"  Start date: {start_date}")
        print(f"  End date: {end_date}")
        print(f"  Duration: {length} days")

        # Get data for the stable period
        stable_period = volatility.loc[str(start_date) : str(end_date)]
        print(f"\n  Stable period statistics:")
        print(f"    Average std: {stable_period['标准差'].mean():.4f}")
        print(f"    Average mean: {stable_period['均值'].mean():.4f}")
        print(f"    Average CV: {stable_period['变异系数'].mean():.4f}")
    else:
        print("\n[NOT FOUND] No stable period found with the given threshold")

    # ====================================================================
    # Step 6: Demonstrate new statistical functions
    # ====================================================================
    print("\n--- Step 6: Using New Statistical Functions ---")

    # Calculate autocorrelation for multiple lags
    print("\nCalculating autocorrelation function (first 10 lags):")
    sample_series = df[field_name].iloc[:1000]  # Use first 1000 points
    acf = calculator.calculate_autocorrelation(sample_series, max_lag=10)

    if not acf.empty:
        print("\nAutocorrelation coefficients:")
        for lag, corr in acf.items():
            print(f"  Lag {lag}: {corr:.4f}")

    # Calculate coefficient of variation for a sample window
    print("\nCalculating coefficient of variation for sample windows:")
    sample_window = df[field_name].iloc[:24]  # One day of hourly data
    cv = calculator._calculate_coefficient_of_variation(sample_window)
    if cv is not None:
        print(f"  CV for first 24 hours: {cv:.4f}")

    # Calculate MAD from mean and median
    mad_mean = calculator._mad_from_mean(sample_window)
    mad_median = calculator._mad_from_median(sample_window)
    print(f"  MAD from mean: {mad_mean:.4f}")
    print(f"  MAD from median: {mad_median:.4f}")

    # ====================================================================
    # Step 7: Save results (optional)
    # ====================================================================
    print("\n--- Step 7: Saving Results (Optional) ---")

    output_path = Path("data/processed/volatility_features_example.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        volatility.to_csv(output_path)
        print(f"[OK] Saved volatility features to: {output_path}")
    except Exception as e:
        print(f"Note: Could not save results: {e}")

    print("\n" + "=" * 70)
    print("VOLATILITY FEATURES EXAMPLE COMPLETED")
    print("=" * 70)


def main():
    """
    Main entry point.
    """
    try:
        example_volatility_features()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n{'=' * 70}")
        print(f"ERROR: {str(e)}")
        print(f"{'=' * 70}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

