"""
Feature Calculator Example Usage

This script demonstrates how to use the FeatureCalculator class directly
and how to interact with the MCP server tools. It includes examples of:
- Statistical features calculation
- Spectral analysis (FFT)
- Volatility features calculation
- Stable period finding
- Single window recalculation
- MCP server tool calls

Requirements:
    - pandas
    - numpy
    - scipy
    - fastmcp (for MCP examples)
"""

from datetime import datetime, timedelta
import json

import numpy as np
import pandas as pd

# Import the FeatureCalculator class
from src.features.calculator import FeatureCalculator
from src.features.mcp_server import (
    _calculate_mean_impl,
    _calculate_statistical_features_impl,
    dataframe_to_json,
    parse_json_to_dataframe,
)


def generate_sample_data(days=30, freq="h"):
    """
    Generate sample time series data for demonstration.

    Creates synthetic temperature and humidity data with:
    - Daily periodic pattern (24-hour cycle)
    - Weekly periodic pattern (7-day cycle)
    - Random noise
    - Trend component

    Args:
        days: Number of days of data to generate
        freq: Frequency of data points ('h' for hourly, '30min' for 30 minutes, etc.)

    Returns:
        DataFrame with DatetimeIndex and temperature/humidity columns
    """
    print(f"\n{'=' * 70}")
    print("GENERATING SAMPLE DATA")
    print(f"{'=' * 70}")

    # Create time index
    start_date = datetime(2024, 1, 1, tzinfo=pd.Timestamp.now().tz)
    periods = days * 24 if freq == "h" else days * 48  # Adjust for frequency
    date_range = pd.date_range(start=start_date, periods=periods, freq=freq)

    # Generate synthetic temperature data with multiple periodic components
    hours = np.arange(len(date_range))

    # Daily cycle (24-hour period) - temperature varies throughout the day
    daily_cycle = 5 * np.sin(2 * np.pi * hours / 24)

    # Weekly cycle (7-day period) - temperature varies throughout the week
    weekly_cycle = 2 * np.sin(2 * np.pi * hours / (24 * 7))

    # Trend component - gradual warming
    trend = 0.01 * hours

    # Random noise
    noise = np.random.normal(0, 1, len(date_range))

    # Combine components
    temperature = 20 + daily_cycle + weekly_cycle + trend + noise

    # Generate humidity data (inversely correlated with temperature)
    humidity = 70 - 0.5 * daily_cycle + np.random.normal(0, 2, len(date_range))

    # Create DataFrame
    df = pd.DataFrame({"空气温度（℃）": temperature, "空气湿度（%）": humidity}, index=date_range)

    print(f"Generated {len(df)} data points from {df.index[0]} to {df.index[-1]}")
    print(
        f"Temperature range: {df['空气温度（℃）'].min():.2f}°C to {df['空气温度（℃）'].max():.2f}°C"
    )
    print(f"Humidity range: {df['空气湿度（%）'].min():.2f}% to {df['空气湿度（%）'].max():.2f}%")

    return df


def example_1_statistical_features():
    """
    Example 1: Calculate statistical features using FeatureCalculator directly.

    Demonstrates:
    - Creating a FeatureCalculator instance
    - Calculating multiple statistical features
    - Automatic range calculation
    - Different resampling frequencies
    """
    print(f"\n{'=' * 70}")
    print("EXAMPLE 1: STATISTICAL FEATURES CALCULATION")
    print(f"{'=' * 70}")

    # Generate sample data
    df = generate_sample_data(days=7, freq="h")

    # Create calculator instance
    calculator = FeatureCalculator()

    # Calculate statistical features with hourly resampling
    print("\n--- Hourly Statistical Features ---")
    features = ["均值", "最大值", "最小值", "标准差", "Q1", "Q3"]
    hourly_stats = calculator.calculate_statistical_features(
        df, field_name="空气温度（℃）", feature_list=features, freq="h"
    )

    print(f"\nCalculated features: {list(hourly_stats.columns)}")
    print(f"Number of time windows: {len(hourly_stats)}")
    print("\nFirst 5 rows:")
    print(hourly_stats.head())

    # Calculate with daily resampling
    print("\n--- Daily Statistical Features ---")
    daily_stats = calculator.calculate_statistical_features(
        df, field_name="空气温度（℃）", feature_list=features, freq="D"
    )

    print(f"\nCalculated features: {list(daily_stats.columns)}")
    print(f"Number of days: {len(daily_stats)}")
    print("\nDaily statistics:")
    print(daily_stats)

    # Note: Range (极差) and rate of change are automatically calculated
    if "极差" in daily_stats.columns:
        print("\nAutomatic range calculation:")
        print(f"  Average daily range: {daily_stats['极差'].mean():.2f}°C")
        print(f"  Maximum daily range: {daily_stats['极差'].max():.2f}°C")


def example_2_spectral_analysis():
    """
    Example 2: Perform spectral analysis (FFT) to identify periodic patterns.

    Demonstrates:
    - FFT analysis on time series data
    - Identifying dominant periods
    - Interpreting spectral results
    """
    print(f"\n{'=' * 70}")
    print("EXAMPLE 2: SPECTRAL ANALYSIS (FFT)")
    print(f"{'=' * 70}")

    # Generate sample data with known periodic patterns
    df = generate_sample_data(days=30, freq="h")

    # Create calculator instance
    calculator = FeatureCalculator()

    # Perform spectral analysis
    print("\n--- Analyzing Temperature Data ---")
    spectrum = calculator.analyze_spectral(df, field_name="空气温度（℃）")

    if spectrum is not None:
        print("\nSpectral analysis completed")
        print(f"Number of frequency components: {len(spectrum)}")

        # Find dominant periods (top 5 by amplitude)
        top_periods = spectrum.nlargest(5, "强度(幅度)")

        print("\nTop 5 dominant periods:")
        print(top_periods.to_string(index=False))

        # Identify the most dominant period
        dominant_period = spectrum.loc[spectrum["强度(幅度)"].idxmax(), "周期(小时)"]
        dominant_amplitude = spectrum.loc[spectrum["强度(幅度)"].idxmax(), "强度(幅度)"]

        print(
            f"\nMost dominant period: {dominant_period:.2f} hours ({dominant_period / 24:.2f} days)"
        )
        print(f"Amplitude: {dominant_amplitude:.4f}")

        # Expected: Should find 24-hour (daily) and 168-hour (weekly) cycles
        print("\nExpected patterns:")
        print("  - Daily cycle: ~24 hours")
        print("  - Weekly cycle: ~168 hours (7 days)")


def example_3_volatility_features():
    """
    Example 3: Calculate volatility features and find stable periods.

    Demonstrates:
    - Daily volatility calculation
    - Mean absolute deviation metrics
    - Autocorrelation analysis
    - Coefficient of variation
    - Finding stable periods
    """
    print(f"\n{'=' * 70}")
    print("EXAMPLE 3: VOLATILITY FEATURES AND STABLE PERIODS")
    print(f"{'=' * 70}")

    # Generate sample data
    df = generate_sample_data(days=30, freq="h")

    # Create calculator instance
    calculator = FeatureCalculator()

    # Calculate volatility features
    print("\n--- Daily Volatility Features ---")
    volatility = calculator.calculate_volatility_features(df, field_name="空气温度（℃）")

    print(f"\nCalculated volatility metrics: {list(volatility.columns)}")
    print(f"Number of days: {len(volatility)}")
    print("\nFirst 10 days:")
    print(volatility.head(10))

    # Analyze volatility statistics
    print("\n--- Volatility Statistics ---")
    print(f"Average daily std: {volatility['标准差'].mean():.4f}°C")
    print(f"Average coefficient of variation: {volatility['变异系数'].mean():.4f}")
    print(f"Average autocorrelation: {volatility['一阶自相关'].mean():.4f}")

    # Find stable periods
    print("\n--- Finding Stable Periods ---")
    threshold = volatility["标准差"].quantile(0.25)  # Use 25th percentile as threshold
    print(f"Stability threshold (25th percentile): {threshold:.4f}°C")

    start_date, end_date, length = calculator.find_stable_period(volatility, threshold)

    if start_date is not None:
        print("\nLongest stable period found:")
        print(f"  Start: {start_date}")
        print(f"  End: {end_date}")
        print(f"  Duration: {length} days")
        print(
            f"  Average std during period: {volatility.loc[str(start_date) : str(end_date), '标准差'].mean():.4f}°C"
        )
    else:
        print("\nNo stable period found with the given threshold")


def example_4_single_window_recalculation():
    """
    Example 4: Recalculate features for a single time window.

    Demonstrates:
    - Single window feature calculation
    - Validation of batch calculations
    - Comparison between batch and single window results
    """
    print(f"\n{'=' * 70}")
    print("EXAMPLE 4: SINGLE WINDOW RECALCULATION")
    print(f"{'=' * 70}")

    # Generate sample data
    df = generate_sample_data(days=3, freq="h")

    # Create calculator instance
    calculator = FeatureCalculator()

    # Calculate features in batch mode
    print("\n--- Batch Calculation (Hourly) ---")
    batch_features = calculator.calculate_statistical_features(
        df,
        field_name="空气温度（℃）",
        feature_list=["均值", "最大值", "最小值", "Q1", "Q3"],
        freq="h",
    )

    print(f"Calculated {len(batch_features)} hourly windows")
    print("\nFirst window (batch mode):")
    print(batch_features.iloc[0])

    # Recalculate for the first window using single window method
    print("\n--- Single Window Recalculation ---")
    first_window_time = batch_features.index[0]
    next_window_time = first_window_time + pd.Timedelta(hours=1)

    # Extract data for the first window
    window_data = df.loc[first_window_time:next_window_time, "空气温度（℃）"]

    print(f"Window time: {first_window_time}")
    print(f"Data points in window: {len(window_data)}")

    # Recalculate features
    single_window_features = calculator.recalculate_single_window(window_data)

    print("\nSingle window features:")
    print(single_window_features)

    # Compare results
    print("\n--- Comparison: Batch vs Single Window ---")
    print(f"Batch 均值: {batch_features.iloc[0]['均值']:.4f}")
    print(f"Single 均值: {single_window_features['均值']:.4f}")
    print(
        f"Difference: {abs(batch_features.iloc[0]['均值'] - single_window_features['均值']):.6f}"
    )


def example_5_mcp_server_calls():
    """
    Example 5: Demonstrate MCP server tool calls.

    Demonstrates:
    - JSON serialization/deserialization
    - Individual feature tool calls
    - Batch feature calculation via MCP
    - Error handling in MCP calls
    """
    print(f"\n{'=' * 70}")
    print("EXAMPLE 5: MCP SERVER TOOL CALLS")
    print(f"{'=' * 70}")

    # Generate sample data
    df = generate_sample_data(days=3, freq="h")

    # Convert DataFrame to JSON format for MCP
    print("\n--- Preparing Data for MCP ---")
    df_reset = df.reset_index()
    df_reset.rename(columns={"index": "time"}, inplace=True)

    data_dict = {"data": df_reset.to_dict(orient="records")}
    data_json = json.dumps(data_dict, default=str, ensure_ascii=False)

    print(f"Serialized {len(df)} data points to JSON")
    print(f"JSON size: {len(data_json)} characters")

    # Example 5a: Call individual MCP tool (calculate_mean)
    print("\n--- Individual MCP Tool: Calculate Mean ---")
    mean_result_json = _calculate_mean_impl(
        data_json=data_json, field_name="空气温度（℃）", freq="D"
    )

    mean_result = json.loads(mean_result_json)
    if "features" in mean_result:
        print("Successfully calculated daily mean")
        print(f"Number of days: {len(mean_result['features'])}")
        print("\nFirst 3 days:")
        for i, day in enumerate(mean_result["features"][:3]):
            print(f"  Day {i + 1}: {day['time'][:10]} - Mean: {day['均值']:.2f}°C")
    else:
        print(f"Error: {mean_result}")

    # Example 5b: Call batch MCP tool (calculate_statistical_features)
    print("\n--- Batch MCP Tool: Calculate Multiple Features ---")
    batch_result_json = _calculate_statistical_features_impl(
        data_json=data_json,
        field_name="空气温度（℃）",
        feature_list=["均值", "最大值", "最小值", "标准差"],
        freq="D",
    )

    batch_result = json.loads(batch_result_json)
    if "features" in batch_result:
        print("Successfully calculated multiple features")
        print(f"Features: {[k for k in batch_result['features'][0].keys() if k != 'time']}")
        print("\nFirst day:")
        first_day = batch_result["features"][0]
        for key, value in first_day.items():
            if key != "time":
                print(f"  {key}: {value:.2f}")
    else:
        print(f"Error: {batch_result}")

    # Example 5c: Error handling - invalid field name
    print("\n--- Error Handling: Invalid Field Name ---")
    error_result_json = _calculate_mean_impl(
        data_json=data_json, field_name="nonexistent_field", freq="D"
    )

    error_result = json.loads(error_result_json)
    if "error" in error_result:
        print(f"Error type: {error_result['error']['type']}")
        print(f"Error message: {error_result['error']['message']}")
        if "details" in error_result["error"]:
            print(f"Error details: {error_result['error']['details']}")

    # Example 5d: JSON round-trip test
    print("\n--- JSON Round-Trip Test ---")
    # Serialize DataFrame to JSON
    json_str = dataframe_to_json(df.head(10))
    print("Serialized 10 rows to JSON")

    # Deserialize back to DataFrame
    df_roundtrip = parse_json_to_dataframe(json_str.replace('"features":', '"data":'))
    print(f"Deserialized back to DataFrame with {len(df_roundtrip)} rows")

    # Compare original and round-trip data
    original_mean = df.head(10)["空气温度（℃）"].mean()
    roundtrip_mean = df_roundtrip["空气温度（℃）"].mean()
    print(f"Original mean: {original_mean:.6f}°C")
    print(f"Round-trip mean: {roundtrip_mean:.6f}°C")
    print(f"Difference: {abs(original_mean - roundtrip_mean):.10f}°C")


def example_6_comprehensive_workflow():
    """
    Example 6: Comprehensive workflow combining all features.

    Demonstrates:
    - Complete analysis pipeline
    - Combining statistical, spectral, and volatility analysis
    - Practical insights from the data
    """
    print(f"\n{'=' * 70}")
    print("EXAMPLE 6: COMPREHENSIVE ANALYSIS WORKFLOW")
    print(f"{'=' * 70}")

    # Generate sample data
    print("\n--- Step 1: Data Preparation ---")
    df = generate_sample_data(days=60, freq="h")

    # Create calculator instance
    calculator = FeatureCalculator()

    # Step 1: Statistical overview
    print("\n--- Step 2: Statistical Overview ---")
    daily_stats = calculator.calculate_statistical_features(
        df,
        field_name="空气温度（℃）",
        feature_list=["均值", "最大值", "最小值", "标准差"],
        freq="D",
    )

    print(f"Daily statistics calculated for {len(daily_stats)} days")
    print(f"Average daily temperature: {daily_stats['均值'].mean():.2f}°C")
    print(
        f"Temperature range: {daily_stats['最小值'].min():.2f}°C to {daily_stats['最大值'].max():.2f}°C"
    )
    print(f"Average daily range: {daily_stats['极差'].mean():.2f}°C")

    # Step 2: Identify periodic patterns
    print("\n--- Step 3: Periodic Pattern Analysis ---")
    spectrum = calculator.analyze_spectral(df, field_name="空气温度（℃）")

    if spectrum is not None:
        top_3_periods = spectrum.nlargest(3, "强度(幅度)")
        print("Top 3 periodic patterns:")
        for idx, row in top_3_periods.iterrows():
            period_hours = row["周期(小时)"]
            amplitude = row["强度(幅度)"]
            print(
                f"  Period: {period_hours:.1f} hours ({period_hours / 24:.1f} days), Amplitude: {amplitude:.4f}"
            )

    # Step 3: Volatility analysis
    print("\n--- Step 4: Volatility Analysis ---")
    volatility = calculator.calculate_volatility_features(df, field_name="空气温度（℃）")

    print(f"Volatility metrics calculated for {len(volatility)} days")
    print(f"Average daily volatility (std): {volatility['标准差'].mean():.4f}°C")
    print(f"Coefficient of variation: {volatility['变异系数'].mean():.4f}")

    # Find most and least volatile days
    most_volatile_day = volatility["标准差"].idxmax()
    least_volatile_day = volatility["标准差"].idxmin()

    print(
        f"\nMost volatile day: {most_volatile_day.date()} (std: {volatility.loc[most_volatile_day, '标准差']:.4f}°C)"
    )
    print(
        f"Least volatile day: {least_volatile_day.date()} (std: {volatility.loc[least_volatile_day, '标准差']:.4f}°C)"
    )

    # Step 4: Find stable periods
    print("\n--- Step 5: Stable Period Identification ---")
    threshold = volatility["标准差"].quantile(0.3)
    start_date, end_date, length = calculator.find_stable_period(volatility, threshold)

    if start_date is not None:
        print(f"Longest stable period (std < {threshold:.4f}°C):")
        print(f"  Duration: {length} days ({start_date} to {end_date})")
        stable_period_data = volatility.loc[str(start_date) : str(end_date)]
        print(f"  Average std: {stable_period_data['标准差'].mean():.4f}°C")
        print(f"  Average temperature: {stable_period_data['均值'].mean():.2f}°C")

    print("\n--- Analysis Complete ---")
    print("This workflow demonstrates how to combine multiple feature calculation")
    print("methods to gain comprehensive insights into time series data.")


def main():
    """
    Main function to run all examples.
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "FEATURE CALCULATOR EXAMPLES")
    print("=" * 70)
    print("\nThis script demonstrates the FeatureCalculator class and MCP server tools.")
    print("It includes examples of statistical features, spectral analysis, volatility")
    print("metrics, and MCP server interactions.")

    try:
        # Run all examples
        example_1_statistical_features()
        example_2_spectral_analysis()
        example_3_volatility_features()
        example_4_single_window_recalculation()
        example_5_mcp_server_calls()
        example_6_comprehensive_workflow()

        print(f"\n{'=' * 70}")
        print(" " * 20 + "ALL EXAMPLES COMPLETED")
        print(f"{'=' * 70}\n")

    except Exception as e:
        print(f"\n{'=' * 70}")
        print(f"ERROR: {str(e)}")
        print(f"{'=' * 70}\n")
        raise


if __name__ == "__main__":
    main()
