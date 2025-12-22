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

# from src.features.mcp_server import (
#     _calculate_mean_impl,
#     _calculate_statistical_features_impl,
#     dataframe_to_json,
#     parse_json_to_dataframe,
# )


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
    features = ["均值", "最大值"]  # , "最小值", "标准差", "Q1", "Q3"
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



def main():
    """
    Main function to run all examples.
    """

    try:
        example_1_statistical_features()

    except Exception as e:
        print(f"\n{'=' * 70}")
        print(f"ERROR: {str(e)}")
        print(f"{'=' * 70}\n")
        raise


if __name__ == "__main__":
    main()
