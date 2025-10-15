import os

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# --- Configuration ---
TARGET_COLUMN = "空气湿度（%）"  # e.g., "空气湿度（%）" or "空气温度（℃）"

# File paths
DATA_PATH = os.path.join("data", "processed", "preprocessed_data.parquet")
OUTPUT_DIR = os.path.join("reports", "figures")

# Date ranges and split point
START_DATE = "2021-01-01"
END_DATE = "2025-01-01"
TRAIN_TEST_SPLIT_DATE = "2024-09-01"


def run_analysis():
    """
    Performs a comprehensive time series analysis including data loading,
    stationarity testing, and generating plots for ARIMA parameter selection.
    """
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        print(f"Error: Target column '{TARGET_COLUMN}' not found in the dataframe.")
        print(f"Available columns are: {df.columns.tolist()}")
        return

    # Ensure index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        # Assuming the index is the time column. If not, you might need to set it, e.g.,
        # df['timestamp'] = pd.to_datetime(df['timestamp'])
        # df.set_index('timestamp', inplace=True)
        print("Converting index to DatetimeIndex.")
        df.index = pd.to_datetime(df.index)

    print(f"Filtering data from {START_DATE} to {END_DATE}...")
    series = df.loc[START_DATE:END_DATE, TARGET_COLUMN]

    print("Resampling data to daily frequency (using mean)...")
    daily_series = series.resample("D").mean().dropna()

    # --- 2. Split Data ---
    train_data = daily_series[:TRAIN_TEST_SPLIT_DATE]
    # test_data = daily_series[TRAIN_TEST_SPLIT_DATE:] # This will be used later for prediction

    print(f"Data split into training set ({len(train_data)} points) and test set.")

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 3. Plot Initial Training Data ---
    print("Plotting the training data...")
    plt.figure(figsize=(14, 7))
    plt.plot(train_data)
    plt.title(f"Time Series of {TARGET_COLUMN} (Training Set)")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(True)
    initial_plot_path = os.path.join(OUTPUT_DIR, "initial_training_series.png")
    plt.savefig(initial_plot_path)
    plt.close()
    print(f"Saved initial training data plot to: {initial_plot_path}")
    print("Please examine this plot to visually assess trend and seasonality.")

    # --- 4. Check for Stationarity (ADF Test) ---
    print("\n--- Stationarity Check (ADF Test) ---")
    adf_result = adfuller(train_data)
    print(f"ADF Statistic: {adf_result[0]}")
    print(f"p-value: {adf_result[1]}")

    data_for_acf_pacf = train_data
    d = 0
    if adf_result[1] > 0.05:
        print("Result: The series is likely non-stationary (p-value > 0.05).")
        print("Applying first-order differencing.")
        d = 1
        data_for_acf_pacf = train_data.diff().dropna()

        # Plot differenced data
        plt.figure(figsize=(14, 7))
        plt.plot(data_for_acf_pacf)
        plt.title(f"First-Order Differenced Time Series (d={d})")
        plt.xlabel("Date")
        plt.ylabel("Differenced Value")
        plt.grid(True)
        diff_plot_path = os.path.join(OUTPUT_DIR, "differenced_series.png")
        plt.savefig(diff_plot_path)
        plt.close()
        print(f"Saved differenced series plot to: {diff_plot_path}")
    else:
        print(
            "Result: The series is likely stationary (p-value <= 0.05). No differencing needed (d=0)."
        )

    # --- 5. Plot ACF and PACF ---
    print("\n--- ACF and PACF Analysis ---")
    print("Generating ACF and PACF plots to determine ARIMA(p,d,q) parameters...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    plot_acf(data_for_acf_pacf, ax=ax1, lags=40)
    ax1.set_title("Autocorrelation Function (ACF)")
    ax1.grid(True)

    plot_pacf(data_for_acf_pacf, ax=ax2, lags=40, method="ywm")
    ax2.set_title("Partial Autocorrelation Function (PACF)")
    ax2.grid(True)

    plt.tight_layout()
    acf_pacf_plot_path = os.path.join(OUTPUT_DIR, "acf_pacf_plots.png")
    plt.savefig(acf_pacf_plot_path)
    plt.close()
    print(f"Saved ACF/PACF plots to: {acf_pacf_plot_path}")
    print("\nInstructions:")
    print("1. Open the 'acf_pacf_plots.png' file.")
    print(
        "2. In the PACF plot, find the lag where the plot first cuts off (drops to zero). This is your suggested 'p' value."
    )
    print(
        "3. In the ACF plot, find the lag where the plot first cuts off. This is your suggested 'q' value."
    )
    print(f"4. The suggested differencing order 'd' is {d}.")
    print("5. Please provide the chosen (p, q) values for the next step.")


if __name__ == "__main__":
    # Before running, ensure you have the necessary libraries installed:
    # pip install pandas pyarrow matplotlib statsmodels
    run_analysis()
