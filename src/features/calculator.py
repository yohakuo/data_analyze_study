"""
Feature Calculator Module

This module provides a unified interface for calculating statistical, spectral,
and volatility features from time series data. The FeatureCalculator class
encapsulates all feature calculation methods and can be exposed via MCP.
"""

from datetime import date
import logging
from typing import Callable

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import detrend

# Configure module logger
logger = logging.getLogger(__name__)


class FeatureCalculator:
    """
    Unified feature calculator for time series data.

    This class provides methods for computing statistical features,
    spectral analysis (FFT), and volatility metrics from pandas DataFrames
    containing time series data.

    Attributes:
        _feature_calculators: Dictionary mapping Chinese feature names to
                            calculation functions or pandas aggregation strings.

    Example:
        >>> calculator = FeatureCalculator()
        >>> df = pd.DataFrame(...)  # Time series data with DatetimeIndex
        >>> stats = calculator.calculate_statis  tical_features(
        ...     df, "temperature", ["均值", "最大值"], freq="h"
        ... )
    """

    def __init__(self):
        """
        Initialize the feature calculator.

        Sets up the internal dictionary mapping feature names to their
        corresponding calculation functions. This includes both simple
        aggregations (mean, max, min) and custom calculations (percent above Q3).
        """
        self._feature_calculators: dict[str, str | Callable] = {
            "均值": self._calculate_mean,
            "中位数": self._calculate_median,
            "最大值": self._calculate_max,
            "最小值": self._calculate_min,
            "标准差": self._calculate_std,
            "Q1": self._calculate_q1,
            "Q3": self._calculate_q3,
            "P10": self._calculate_p10,
            "超过Q3占时比": self._calculate_percent_above_q3,
        }

    def calculate_statistical_features(
        self, df: pd.DataFrame, field_name: str, feature_list: list[str], freq: str = "h"
    ) -> pd.DataFrame:
        """
        Calculate statistical features from time series data.

        This method resamples the time series at the specified frequency and
        computes the requested statistical features. It automatically calculates
        range (极差) when both max and min are requested, and computes the rate
        of change for the range.

        Args:
            df: DataFrame with DatetimeIndex containing time series data
            field_name: Name of the column to calculate features for
            feature_list: List of feature names to calculate. Valid options:
                         均值, 中位数, 最大值, 最小值, 标准差, Q1, Q3, P10, 超过Q3占时比
            freq: Resampling frequency. Options:
                  'h' - hourly, 'D' - daily, 'W' - weekly, 'M' - monthly

        Returns:
            DataFrame with calculated features indexed by time. Returns empty
            DataFrame if input is invalid.

        Raises:
            ValueError: If df is empty or field_name not in columns

        Example:
            >>> calculator = FeatureCalculator()
            >>> features = calculator.calculate_statistical_features(
            ...     df, "temperature", ["均值", "最大值", "最小值"], freq="D"
            ... )
        """
        # Input validation
        if df.empty:
            logger.error(
                "Empty DataFrame provided to calculate_statistical_features",
                extra={"field_name": field_name, "freq": freq},
            )
            return pd.DataFrame()

        if field_name not in df.columns:
            logger.error(
                f"Field '{field_name}' not found in DataFrame",
                extra={
                    "field_name": field_name,
                    "available_columns": list(df.columns),
                    "freq": freq,
                },
            )
            return pd.DataFrame()

        # Filter requested features to only those available
        agg_functions = {
            feature: self._feature_calculators[feature]
            for feature in feature_list
            if feature in self._feature_calculators
        }

        if not agg_functions:
            logger.error(
                "No valid features found in feature_list",
                extra={
                    "field_name": field_name,
                    "requested_features": feature_list,
                    "available_features": list(self._feature_calculators.keys()),
                },
            )
            return pd.DataFrame()

        logger.info(
            f"Calculating statistical features for field '{field_name}'",
            extra={
                "field_name": field_name,
                "features": list(agg_functions.keys()),
                "freq": freq,
                "data_shape": df.shape,
            },
        )

        # Resample and calculate all features at once using .agg()
        resampled_stats = df[field_name].resample(freq).agg(**agg_functions)

        # Automatic range calculation when both max and min are requested
        if "最大值" in resampled_stats.columns and "最小值" in resampled_stats.columns:
            resampled_stats["极差"] = resampled_stats["最大值"] - resampled_stats["最小值"]

        # Calculate rate of change for range
        if "极差" in resampled_stats.columns:
            resampled_stats["极差的时间变化率"] = resampled_stats["极差"].pct_change().fillna(0)

        # Rename median column to include (Q2) notation
        if "中位数" in resampled_stats.columns:
            resampled_stats.rename(columns={"中位数": "中位数 (Q2)"}, inplace=True)

        return resampled_stats

    def analyze_spectral(self, df: pd.DataFrame, field_name: str) -> pd.DataFrame | None:
        """
        Perform FFT spectral analysis on time series data.

        This method removes the long-term trend from the data and applies
        Fast Fourier Transform to identify dominant periodic patterns.
        Frequencies are converted to periods in hours for interpretability.

        Args:
            df: DataFrame with DatetimeIndex containing time series data
            field_name: Name of the column to analyze

        Returns:
            DataFrame with two columns:
                - 周期(小时): Periods in hours
                - 强度(幅度): Corresponding amplitudes
            Returns None if df is empty.

        Example:
            >>> calculator = FeatureCalculator()
            >>> spectrum = calculator.analyze_spectral(df, "temperature")
            >>> # Find dominant period
            >>> dominant_period = spectrum.loc[spectrum["强度(幅度)"].idxmax(), "周期(小时)"]
        """
        if df.empty:
            logger.warning(
                "Empty DataFrame provided to analyze_spectral", extra={"field_name": field_name}
            )
            return None

        if field_name not in df.columns:
            logger.error(
                f"Field '{field_name}' not found in DataFrame for spectral analysis",
                extra={"field_name": field_name, "available_columns": list(df.columns)},
            )
            return None

        logger.info(
            f"Starting spectral analysis (FFT) for field '{field_name}'",
            extra={"field_name": field_name, "data_points": len(df)},
        )

        # 1. Remove long-term trend
        logger.debug("Detrending data for spectral analysis")
        detrended_values = detrend(df[field_name].values)

        # 2. Compute FFT
        N = len(detrended_values)
        # Automatically calculate sampling interval from DataFrame index
        T = (df.index[1] - df.index[0]).total_seconds()
        yf = fft(detrended_values)
        xf = fftfreq(N, T)[: N // 2]
        amplitude = 2.0 / N * np.abs(yf[0 : N // 2])

        # 3. Convert frequencies to periods in hours
        periods_in_hours = np.full_like(xf, np.inf)
        non_zero_indices = xf > 0
        periods_in_hours[non_zero_indices] = 1 / xf[non_zero_indices] / 3600

        # 4. Build result DataFrame
        spectrum_df = pd.DataFrame({"周期(小时)": periods_in_hours, "强度(幅度)": amplitude})

        logger.info(
            f"Spectral analysis completed for field '{field_name}'",
            extra={"field_name": field_name, "spectrum_points": len(spectrum_df)},
        )

        return spectrum_df

    def calculate_volatility_features(self, df: pd.DataFrame, field_name: str) -> pd.DataFrame:
        """
        Calculate daily volatility features.

        This method resamples data to daily frequency and computes various
        volatility metrics including standard deviation, mean absolute deviations,
        autocorrelation, and coefficient of variation.

        Args:
            df: DataFrame with DatetimeIndex containing time series data
            field_name: Name of the column to calculate volatility for

        Returns:
            DataFrame with daily volatility metrics:
                - 均值: Daily mean
                - 标准差: Daily standard deviation
                - 平均绝对偏差_均值: MAD from mean
                - 平均绝对偏差_中位数: MAD from median
                - 一阶自相关: First-order autocorrelation
                - 变异系数: Coefficient of variation (std/mean)
            Rows with NaN values are removed.

        Example:
            >>> calculator = FeatureCalculator()
            >>> volatility = calculator.calculate_volatility_features(df, "temperature")
            >>> # Find most stable day
            >>> most_stable = volatility.loc[volatility["标准差"].idxmin()]
        """
        if df.empty:
            logger.error(
                "Empty DataFrame provided to calculate_volatility_features",
                extra={"field_name": field_name},
            )
            return pd.DataFrame()

        if field_name not in df.columns:
            logger.error(
                f"Field '{field_name}' not found in DataFrame for volatility calculation",
                extra={"field_name": field_name, "available_columns": list(df.columns)},
            )
            return pd.DataFrame()

        logger.info(
            f"Calculating daily volatility features for field '{field_name}'",
            extra={"field_name": field_name, "data_points": len(df)},
        )

        # Resample to daily frequency and calculate volatility metrics
        daily_stats = (
            df[field_name]
            .resample("D")
            .agg(
                均值="mean",
                标准差="std",
                平均绝对偏差_均值=self._mad_from_mean,
                平均绝对偏差_中位数=self._mad_from_median,
                一阶自相关=self._autocorr_lag1,
            )
        )

        # Calculate coefficient of variation
        daily_stats["变异系数"] = daily_stats["标准差"] / daily_stats["均值"]

        # Remove rows with NaN values
        rows_before = len(daily_stats)
        daily_stats.dropna(inplace=True)
        rows_after = len(daily_stats)

        if rows_before > rows_after:
            logger.debug(
                f"Removed {rows_before - rows_after} rows with NaN values from volatility features",
                extra={"rows_removed": rows_before - rows_after},
            )

        logger.info(
            f"Daily volatility calculation completed for field '{field_name}'",
            extra={"field_name": field_name, "result_rows": len(daily_stats)},
        )
        return daily_stats

    def find_stable_period(
        self, volatility_df: pd.DataFrame, threshold: float
    ) -> tuple[date | None, date | None, int]:
        """
        Find the longest consecutive stable period.

        A stable period is defined as consecutive days where the standard
        deviation remains below the specified threshold.

        Args:
            volatility_df: DataFrame from calculate_volatility_features,
                          must contain a '标准差' column
            threshold: Standard deviation threshold for stability

        Returns:
            Tuple of (start_date, end_date, length_in_days):
                - start_date: First date of the longest stable period
                - end_date: Last date of the longest stable period
                - length_in_days: Number of consecutive stable days
            Returns (None, None, 0) if no stable period exists.

        Example:
            >>> calculator = FeatureCalculator()
            >>> volatility = calculator.calculate_volatility_features(df, "temperature")
            >>> start, end, length = calculator.find_stable_period(volatility, threshold=1.5)
            >>> print(f"Stable period: {start} to {end} ({length} days)")
        """
        logger.info(
            f"Finding stable period with threshold {threshold}",
            extra={"threshold": threshold, "total_days": len(volatility_df)},
        )

        volatility_df["is_stable"] = volatility_df["标准差"] < threshold

        # Use groupby and shift to identify consecutive stable periods
        streaks = (
            volatility_df[volatility_df["is_stable"]]
            .groupby((volatility_df["is_stable"] != volatility_df["is_stable"].shift()).cumsum())
            .size()
        )

        if streaks.empty:
            logger.info("No stable period found", extra={"threshold": threshold})
            return None, None, 0

        # Find the longest streak
        longest_streak_len = streaks.max()
        streak_group_id = streaks.idxmax()

        # Extract dates for the longest streak
        streak_df = volatility_df[
            (volatility_df["is_stable"])
            & (
                (volatility_df["is_stable"] != volatility_df["is_stable"].shift()).cumsum()
                == streak_group_id
            )
        ]

        start_date = streak_df.index.min().date()
        end_date = streak_df.index.max().date()

        logger.info(
            f"Found stable period: {start_date} to {end_date} ({longest_streak_len} days)",
            extra={
                "start_date": str(start_date),
                "end_date": str(end_date),
                "length_days": longest_streak_len,
                "threshold": threshold,
            },
        )

        return start_date, end_date, longest_streak_len

    def recalculate_single_window(self, series: pd.Series) -> pd.Series:
        """
        Recalculate features for a single time window.

        This method computes all non-rate-of-change statistical features
        for a single time window (e.g., one hour of data). Useful for
        validation and testing purposes.

        Args:
            series: Time series data for one window

        Returns:
            Series with calculated feature values, indexed by feature names.
            Returns empty Series if input is empty.

        Example:
            >>> calculator = FeatureCalculator()
            >>> # Get one hour of data
            >>> hour_data = df.loc["2024-01-01 00:00":"2024-01-01 01:00", "temperature"]
            >>> features = calculator.recalculate_single_window(hour_data)
            >>> print(features["均值"])
        """
        if series.empty:
            logger.warning("Empty series provided to recalculate_single_window")
            return pd.Series(dtype="object")

        logger.debug(
            "Recalculating features for single window", extra={"data_points": len(series)}
        )

        recalculated_features = pd.Series(
            {
                "均值": series.mean(),
                "中位数_Q2": series.median(),
                "最大值": series.max(),
                "最小值": series.min(),
                "Q1": series.quantile(0.25),
                "Q3": series.quantile(0.75),
                "P10": series.quantile(0.10),
                "极差": series.max() - series.min(),
                "超过Q3占时比": self._calculate_percent_above_q3(series),
            }
        )

        return recalculated_features

    # ====================================================================
    #   内部函数
    # ====================================================================
    def _calculate_mean(self, series: pd.Series) -> float:
        """
        Calculate the mean of a series.

        Args:
            series: Time series data

        Returns:
            Mean value of the series
        """
        return series.mean()

    def _calculate_median(self, series: pd.Series) -> float:
        """
        Calculate the median of a series.

        Args:
            series: Time series data

        Returns:
            Median value of the series
        """
        return series.median()

    def _calculate_max(self, series: pd.Series) -> float:
        """
        Calculate the maximum value of a series.

        Args:
            series: Time series data

        Returns:
            Maximum value of the series
        """
        return series.max()

    def _calculate_min(self, series: pd.Series) -> float:
        """
        Calculate the minimum value of a series.

        Args:
            series: Time series data

        Returns:
            Minimum value of the series
        """
        return series.min()

    def _calculate_std(self, series: pd.Series) -> float:
        """
        Calculate the standard deviation of a series.

        Args:
            series: Time series data

        Returns:
            Standard deviation of the series
        """
        return series.std()

    def _calculate_q1(self, series: pd.Series) -> float:
        """
        Calculate the first quartile (Q1) of a series.

        Args:
            series: Time series data

        Returns:
            First quartile (Q1) value of the series
        """
        return series.quantile(0.25)

    def _calculate_q3(self, series: pd.Series) -> float:
        """
        Calculate the third quartile (Q3) of a series.

        Args:
            series: Time series data

        Returns:
            Third quartile (Q3) value of the series
        """
        return series.quantile(0.75)

    def _calculate_p10(self, series: pd.Series) -> float:
        """
        Calculate the 10th percentile (P10) of a series.

        Args:
            series: Time series data

        Returns:
            10th percentile (P10) value of the series
        """
        return series.quantile(0.10)

    def _calculate_percent_above_q3(self, series: pd.Series) -> float:
        """
        Calculate the percentage of values above the third quartile (Q3).

        Args:
            series: Time series data

        Returns:
            Percentage of values above Q3 (between 0.0 and 1.0)

        Note:
            Returns 0.0 if series has fewer than 4 points or if Q3 equals max.
        """
        if len(series) < 4:
            return 0.0
        q3 = series.quantile(0.75)
        # Avoid case where Q3 equals max value (no values can be above Q3)
        if q3 == series.max():
            return 0.0
        count_above_q3 = (series > q3).sum()
        percent_above_q3 = count_above_q3 / len(series)
        return percent_above_q3

    def _mad_from_mean(self, series: pd.Series) -> float:
        """
        Calculate mean absolute deviation from mean.

        Args:
            series: Time series data

        Returns:
            Mean absolute deviation from the series mean
        """
        return (series - series.mean()).abs().mean()

    def _mad_from_median(self, series: pd.Series) -> float:
        """
        Calculate mean absolute deviation from median.

        Args:
            series: Time series data

        Returns:
            Mean absolute deviation from the series median
        """
        return (series - series.median()).abs().mean()

    def _autocorr_lag1(self, series: pd.Series) -> float | None:
        """
        Calculate first-order autocorrelation.

        Args:
            series: Time series data

        Returns:
            First-order autocorrelation coefficient, or None if series
            has fewer than 2 data points
        """
        if len(series) < 2:
            return None
        return series.autocorr(lag=1)
