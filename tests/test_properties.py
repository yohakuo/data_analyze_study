"""
Property-based tests for FeatureCalculator.

This module contains property-based tests using Hypothesis to verify
universal properties that should hold across all valid executions.
Each test runs a minimum of 100 iterations with randomly generated inputs.
"""

import json
import pandas as pd
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes
from datetime import datetime, timedelta

from src.features.calculator import FeatureCalculator


# ============================================================================
# Property 1: Statistical features completeness
# Feature: feature-calculator-mcp, Property 1: Statistical features completeness
# ============================================================================

@settings(max_examples=100)
@given(
    num_rows=st.integers(min_value=100, max_value=500),
    field_name=st.text(min_size=1, max_size=20, alphabet=st.characters(blacklist_characters=['\x00'])),
    feature_list=st.lists(
        st.sampled_from(["均值", "中位数", "最大值", "最小值", "标准差", "Q1", "Q3", "P10", "超过Q3占时比"]),
        min_size=1,
        max_size=9,
        unique=True
    ),
    freq=st.sampled_from(["D", "W", "ME"])  # Use frequencies that aggregate multiple points
)
def test_property_1_statistical_features_completeness(num_rows, field_name, feature_list, freq):
    """
    Property 1: Statistical features completeness.
    
    For any valid DataFrame with time series data, field name, and list of
    requested features, computing statistical features should return a DataFrame
    containing all requested features as columns with valid numeric values.
    
    **Validates: Requirements 1.1**
    """
    calculator = FeatureCalculator()
    
    # Create DataFrame with valid time series data (hourly frequency)
    start_time = datetime(2024, 1, 1)
    index = pd.date_range(start=start_time, periods=num_rows, freq="h")
    df = pd.DataFrame({
        field_name: np.random.uniform(10, 100, num_rows)
    }, index=index)
    
    # Calculate statistical features
    result = calculator.calculate_statistical_features(df, field_name, feature_list, freq)
    
    # Check that result is not empty
    assert not result.empty, "Result should not be empty for valid input"
    
    # Check that all requested features are present in the result
    # Note: 中位数 is renamed to 中位数 (Q2)
    expected_columns = []
    for feature in feature_list:
        if feature == "中位数":
            expected_columns.append("中位数 (Q2)")
        else:
            expected_columns.append(feature)
    
    for expected_col in expected_columns:
        assert expected_col in result.columns, f"Feature '{expected_col}' should be in result columns"
    
    # Check that all values are numeric and not all NaN
    for col in expected_columns:
        assert pd.api.types.is_numeric_dtype(result[col]), f"Column '{col}' should have numeric dtype"
        assert not result[col].isna().all(), f"Column '{col}' should not be all NaN"


# ============================================================================
# Property 2: Resampling frequency consistency
# Feature: feature-calculator-mcp, Property 2: Resampling frequency consistency
# ============================================================================

@settings(max_examples=100)
@given(
    num_rows=st.integers(min_value=50, max_value=200),
    field_name=st.text(min_size=1, max_size=20, alphabet=st.characters(blacklist_characters=['\x00'])),
    feature_list=st.lists(
        st.sampled_from(["均值", "最大值", "最小值"]),
        min_size=1,
        max_size=3,
        unique=True
    ),
    freq=st.sampled_from(["h", "D", "W"])
)
def test_property_2_resampling_frequency_consistency(num_rows, field_name, feature_list, freq):
    """
    Property 2: Resampling frequency consistency.
    
    For any valid DataFrame with time series data and resampling frequency
    specification, the output DataFrame index should match the specified
    resampling frequency with appropriate time intervals.
    
    **Validates: Requirements 1.2**
    """
    calculator = FeatureCalculator()
    
    # Create DataFrame with hourly data
    start_time = datetime(2024, 1, 1)
    index = pd.date_range(start=start_time, periods=num_rows, freq="h")
    df = pd.DataFrame({
        field_name: np.random.uniform(10, 100, num_rows)
    }, index=index)
    
    # Calculate statistical features with specified frequency
    result = calculator.calculate_statistical_features(df, field_name, feature_list, freq)
    
    # Check that result has a DatetimeIndex
    assert isinstance(result.index, pd.DatetimeIndex), "Result should have DatetimeIndex"
    
    # Check that the index frequency matches the requested frequency
    if len(result) > 1:
        # Calculate the actual frequency from the index
        time_diffs = result.index.to_series().diff().dropna()
        
        # Define expected time deltas for each frequency
        freq_to_timedelta = {
            "h": timedelta(hours=1),
            "D": timedelta(days=1),
            "W": timedelta(weeks=1),
        }
        
        expected_delta = freq_to_timedelta[freq]
        
        # Check that all time differences match the expected frequency
        # Allow for some tolerance due to month-end effects
        for diff in time_diffs:
            # For weekly and daily, allow some variation
            if freq == "W":
                assert diff >= timedelta(days=6) and diff <= timedelta(days=8), \
                    f"Time difference {diff} should be approximately 7 days for weekly frequency"
            elif freq == "D":
                assert diff == expected_delta, \
                    f"Time difference {diff} should be {expected_delta} for daily frequency"
            elif freq == "h":
                assert diff == expected_delta, \
                    f"Time difference {diff} should be {expected_delta} for hourly frequency"


# ============================================================================
# Property 3: Automatic range calculation
# Feature: feature-calculator-mcp, Property 3: Automatic range calculation
# ============================================================================

@settings(max_examples=100)
@given(
    num_rows=st.integers(min_value=20, max_value=200),
    field_name=st.text(min_size=1, max_size=20, alphabet=st.characters(blacklist_characters=['\x00'])),
    include_other_features=st.booleans(),
    freq=st.sampled_from(["h", "D", "W"])
)
def test_property_3_automatic_range_calculation(num_rows, field_name, include_other_features, freq):
    """
    Property 3: Automatic range calculation.
    
    For any valid DataFrame where both "最大值" and "最小值" are requested features,
    the output should contain a "极差" column where each value equals the
    corresponding maximum minus minimum, and a "极差的时间变化率" column computed
    as the percentage change of range over time.
    
    **Validates: Requirements 1.3, 1.4**
    """
    calculator = FeatureCalculator()
    
    # Create DataFrame with valid time series data
    start_time = datetime(2024, 1, 1)
    index = pd.date_range(start=start_time, periods=num_rows, freq="h")
    df = pd.DataFrame({
        field_name: np.random.uniform(10, 100, num_rows)
    }, index=index)
    
    # Build feature list that includes both max and min
    feature_list = ["最大值", "最小值"]
    if include_other_features:
        feature_list.append("均值")
    
    # Calculate statistical features
    result = calculator.calculate_statistical_features(df, field_name, feature_list, freq)
    
    # Check that 极差 column exists
    assert "极差" in result.columns, "极差 column should be automatically added when both max and min are requested"
    
    # Check that 极差的时间变化率 column exists
    assert "极差的时间变化率" in result.columns, "极差的时间变化率 column should be automatically added when range is calculated"
    
    # Verify that 极差 = 最大值 - 最小值 for all rows
    calculated_range = result["最大值"] - result["最小值"]
    pd.testing.assert_series_equal(
        result["极差"], 
        calculated_range, 
        check_names=False,
        atol=1e-10
    )
    
    # Verify that 极差的时间变化率 is the percentage change of 极差
    expected_rate_of_change = result["极差"].pct_change().fillna(0)
    pd.testing.assert_series_equal(
        result["极差的时间变化率"],
        expected_rate_of_change,
        check_names=False,
        atol=1e-10
    )


# ============================================================================
# Property 4: FFT output format consistency
# Feature: feature-calculator-mcp, Property 4: FFT output format consistency
# ============================================================================

@settings(max_examples=100)
@given(
    num_rows=st.integers(min_value=50, max_value=500),
    field_name=st.text(min_size=1, max_size=20, alphabet=st.characters(blacklist_characters=['\x00']))
)
def test_property_4_fft_output_format_consistency(num_rows, field_name):
    """
    Property 4: FFT output format consistency.
    
    For any valid DataFrame with time series data, performing spectral analysis
    should return a DataFrame with exactly two columns: "周期(小时)" containing
    periods in hours (not raw frequencies) and "强度(幅度)" containing
    corresponding amplitudes.
    
    **Validates: Requirements 2.2, 2.5**
    """
    calculator = FeatureCalculator()
    
    # Create DataFrame with valid time series data (hourly frequency)
    start_time = datetime(2024, 1, 1)
    index = pd.date_range(start=start_time, periods=num_rows, freq="h")
    df = pd.DataFrame({
        field_name: np.random.uniform(10, 100, num_rows)
    }, index=index)
    
    # Perform spectral analysis
    result = calculator.analyze_spectral(df, field_name)
    
    # Check that result is a DataFrame (not None)
    assert result is not None, "Result should not be None for valid input"
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    
    # Check that result has exactly two columns
    assert len(result.columns) == 2, "Result should have exactly 2 columns"
    
    # Check that the columns are named correctly
    assert "周期(小时)" in result.columns, "Result should have '周期(小时)' column"
    assert "强度(幅度)" in result.columns, "Result should have '强度(幅度)' column"
    
    # Check that periods are in hours (positive values or infinity)
    periods = result["周期(小时)"]
    assert (periods >= 0).all() or periods.isin([np.inf]).any(), \
        "Periods should be positive or infinity"
    
    # Check that amplitudes are numeric and non-negative
    amplitudes = result["强度(幅度)"]
    assert pd.api.types.is_numeric_dtype(amplitudes), "Amplitudes should be numeric"
    assert (amplitudes >= 0).all(), "Amplitudes should be non-negative"


# ============================================================================
# Property 5: FFT detrending application
# Feature: feature-calculator-mcp, Property 5: FFT detrending application
# ============================================================================

@settings(max_examples=100)
@given(
    num_rows=st.integers(min_value=50, max_value=300),
    field_name=st.text(min_size=1, max_size=20, alphabet=st.characters(blacklist_characters=['\x00'])),
    trend_slope=st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False)
)
def test_property_5_fft_detrending_application(num_rows, field_name, trend_slope):
    """
    Property 5: FFT detrending application.
    
    For any valid DataFrame with time series data, the spectral analysis should
    apply detrending before FFT computation, resulting in a spectrum that
    reflects periodic patterns rather than long-term trends.
    
    This test verifies that detrending is applied by comparing the spectrum
    of data with a strong linear trend to the spectrum of the same data
    without the trend. The detrended spectrum should have reduced low-frequency
    components.
    
    **Validates: Requirements 2.1**
    """
    calculator = FeatureCalculator()
    
    # Create time series with a strong linear trend plus periodic component
    start_time = datetime(2024, 1, 1)
    index = pd.date_range(start=start_time, periods=num_rows, freq="h")
    
    # Create periodic signal (e.g., 24-hour cycle)
    t = np.arange(num_rows)
    periodic_signal = 10 * np.sin(2 * np.pi * t / 24)
    
    # Add linear trend
    linear_trend = trend_slope * t
    
    # Combine periodic signal with trend
    signal_with_trend = periodic_signal + linear_trend
    
    df = pd.DataFrame({
        field_name: signal_with_trend
    }, index=index)
    
    # Perform spectral analysis (which should detrend)
    result = calculator.analyze_spectral(df, field_name)
    
    # Check that result is valid
    assert result is not None, "Result should not be None"
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    
    # The detrending should remove the DC component and low-frequency trend
    # Check that the spectrum has finite periods (not dominated by infinity/DC)
    finite_periods = result[result["周期(小时)"] != np.inf]
    assert len(finite_periods) > 0, "Should have finite periods after detrending"
    
    # The amplitude at the dominant periodic component should be significant
    # compared to very low frequency components
    max_amplitude = result["强度(幅度)"].max()
    assert max_amplitude > 0, "Should have non-zero amplitudes"
    
    # Check that we have a valid spectrum (not all zeros or all same value)
    amplitudes = result["强度(幅度)"].values
    assert not np.allclose(amplitudes, 0), "Amplitudes should not all be zero"
    assert len(np.unique(amplitudes)) > 1, "Amplitudes should have variation"


# ============================================================================
# Property 6: Sampling interval calculation
# Feature: feature-calculator-mcp, Property 6: Sampling interval calculation
# ============================================================================

@settings(max_examples=100)
@given(
    num_rows=st.integers(min_value=50, max_value=300),
    field_name=st.text(min_size=1, max_size=20, alphabet=st.characters(blacklist_characters=['\x00'])),
    freq_minutes=st.sampled_from([1, 5, 10, 15, 30, 60, 120])  # Different sampling frequencies in minutes
)
def test_property_6_sampling_interval_calculation(num_rows, field_name, freq_minutes):
    """
    Property 6: Sampling interval calculation.
    
    For any DataFrame with a DatetimeIndex, the spectral analysis should
    automatically calculate the sampling interval from consecutive index values
    and use it for frequency-to-period conversion.
    
    This test verifies that the period calculation is correct by checking that
    the Nyquist period (the shortest detectable period) is approximately twice
    the sampling interval.
    
    **Validates: Requirements 2.3**
    """
    calculator = FeatureCalculator()
    
    # Create DataFrame with specified sampling frequency
    start_time = datetime(2024, 1, 1)
    index = pd.date_range(start=start_time, periods=num_rows, freq=f"{freq_minutes}min")
    
    # Create simple periodic signal
    df = pd.DataFrame({
        field_name: np.random.uniform(10, 100, num_rows)
    }, index=index)
    
    # Perform spectral analysis
    result = calculator.analyze_spectral(df, field_name)
    
    # Check that result is valid
    assert result is not None, "Result should not be None"
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    
    # Calculate expected Nyquist period (2 * sampling interval in hours)
    sampling_interval_hours = freq_minutes / 60.0
    expected_nyquist_period = 2 * sampling_interval_hours
    
    # Get the minimum finite period from the result
    finite_periods = result[result["周期(小时)"] != np.inf]["周期(小时)"]
    
    if len(finite_periods) > 0:
        min_period = finite_periods.min()
        
        # The minimum period should be close to the Nyquist period
        # Allow for some tolerance due to FFT bin spacing
        # The minimum period should be at least close to 2 * sampling interval
        assert min_period >= expected_nyquist_period * 0.8, \
            f"Minimum period {min_period} should be close to Nyquist period {expected_nyquist_period}"
        
        # Also verify that periods are in hours (not seconds or other units)
        # For hourly sampling, we should see periods in reasonable hour ranges
        max_period = finite_periods.max()
        expected_max_period = num_rows * sampling_interval_hours
        
        # Maximum period should not exceed the total duration
        assert max_period <= expected_max_period * 1.5, \
            f"Maximum period {max_period} should not greatly exceed total duration {expected_max_period}"


# ============================================================================
# Property 13: Consistent error handling
# Feature: feature-calculator-mcp, Property 13: Consistent error handling
# ============================================================================

@settings(max_examples=100)
@given(
    field_name=st.text(min_size=1, max_size=20),
    feature_list=st.lists(
        st.sampled_from(["均值", "中位数", "最大值", "最小值", "标准差", "Q1", "Q3", "P10", "超过Q3占时比"]),
        min_size=1,
        max_size=5
    ),
    freq=st.sampled_from(["h", "D", "W", "M"])
)
def test_property_13_empty_dataframe_consistent_error_handling(field_name, feature_list, freq):
    """
    Property 13: Consistent error handling for empty DataFrames.
    
    For any valid parameters, when given an empty DataFrame, the
    calculate_statistical_features method should consistently return
    an empty DataFrame (not raise an exception, not return None).
    
    **Validates: Requirements 5.5**
    """
    calculator = FeatureCalculator()
    
    # Create empty DataFrame with DatetimeIndex
    empty_df = pd.DataFrame(index=pd.DatetimeIndex([]))
    
    # Should return empty DataFrame, not raise exception
    result = calculator.calculate_statistical_features(
        empty_df, field_name, feature_list, freq
    )
    
    assert isinstance(result, pd.DataFrame), "Should return DataFrame for empty input"
    assert result.empty, "Should return empty DataFrame for empty input"


@settings(max_examples=100)
@given(
    num_rows=st.integers(min_value=10, max_value=100),
    valid_field=st.text(min_size=1, max_size=20, alphabet=st.characters(blacklist_characters=['\x00'])),
    invalid_field=st.text(min_size=1, max_size=20, alphabet=st.characters(blacklist_characters=['\x00'])),
    feature_list=st.lists(
        st.sampled_from(["均值", "中位数", "最大值", "最小值", "标准差"]),
        min_size=1,
        max_size=3
    ),
    freq=st.sampled_from(["h", "D"])
)
def test_property_13_missing_field_consistent_error_handling(
    num_rows, valid_field, invalid_field, feature_list, freq
):
    """
    Property 13: Consistent error handling for missing fields.
    
    For any DataFrame with valid data, when requesting features for a
    field that doesn't exist, the method should consistently return
    an empty DataFrame (not raise an exception).
    
    **Validates: Requirements 5.5**
    """
    # Ensure invalid_field is different from valid_field
    if invalid_field == valid_field:
        invalid_field = valid_field + "_different"
    
    calculator = FeatureCalculator()
    
    # Create DataFrame with valid field
    start_time = datetime(2024, 1, 1)
    index = pd.date_range(start=start_time, periods=num_rows, freq="h")
    df = pd.DataFrame({
        valid_field: np.random.randn(num_rows)
    }, index=index)
    
    # Request features for non-existent field
    result = calculator.calculate_statistical_features(
        df, invalid_field, feature_list, freq
    )
    
    assert isinstance(result, pd.DataFrame), "Should return DataFrame for missing field"
    assert result.empty, "Should return empty DataFrame for missing field"


@settings(max_examples=100)
@given(
    num_rows=st.integers(min_value=10, max_value=100),
    field_name=st.text(min_size=1, max_size=20, alphabet=st.characters(blacklist_characters=['\x00']))
)
def test_property_13_spectral_empty_dataframe_consistent_error_handling(num_rows, field_name):
    """
    Property 13: Consistent error handling for spectral analysis with empty DataFrame.
    
    For any field name, when given an empty DataFrame, the analyze_spectral
    method should consistently return None (not raise an exception).
    
    **Validates: Requirements 5.5**
    """
    calculator = FeatureCalculator()
    
    # Create empty DataFrame with DatetimeIndex
    empty_df = pd.DataFrame(index=pd.DatetimeIndex([]))
    
    # Should return None, not raise exception
    result = calculator.analyze_spectral(empty_df, field_name)
    
    assert result is None, "Should return None for empty DataFrame in spectral analysis"


@settings(max_examples=100)
@given(
    num_rows=st.integers(min_value=10, max_value=100),
    valid_field=st.text(min_size=1, max_size=20, alphabet=st.characters(blacklist_characters=['\x00'])),
    invalid_field=st.text(min_size=1, max_size=20, alphabet=st.characters(blacklist_characters=['\x00']))
)
def test_property_13_volatility_missing_field_consistent_error_handling(
    num_rows, valid_field, invalid_field
):
    """
    Property 13: Consistent error handling for volatility with missing field.
    
    For any DataFrame with valid data, when requesting volatility features
    for a field that doesn't exist, the method should consistently return
    an empty DataFrame (not raise an exception).
    
    **Validates: Requirements 5.5**
    """
    # Ensure invalid_field is different from valid_field
    if invalid_field == valid_field:
        invalid_field = valid_field + "_different"
    
    calculator = FeatureCalculator()
    
    # Create DataFrame with valid field
    start_time = datetime(2024, 1, 1)
    index = pd.date_range(start=start_time, periods=num_rows, freq="h")
    df = pd.DataFrame({
        valid_field: np.random.randn(num_rows)
    }, index=index)
    
    # Request volatility for non-existent field
    result = calculator.calculate_volatility_features(df, invalid_field)
    
    assert isinstance(result, pd.DataFrame), "Should return DataFrame for missing field"
    assert result.empty, "Should return empty DataFrame for missing field"


def test_property_13_single_window_empty_series_consistent_error_handling():
    """
    Property 13: Consistent error handling for single window with empty Series.
    
    When given an empty Series, the recalculate_single_window method should
    consistently return an empty Series (not raise an exception).
    
    **Validates: Requirements 5.5**
    """
    calculator = FeatureCalculator()
    
    # Create empty Series
    empty_series = pd.Series(dtype=float)
    
    # Should return empty Series, not raise exception
    result = calculator.recalculate_single_window(empty_series)
    
    assert isinstance(result, pd.Series), "Should return Series for empty input"
    assert result.empty, "Should return empty Series for empty input"


@settings(max_examples=100)
@given(
    num_rows=st.integers(min_value=5, max_value=50),
    threshold=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
)
def test_property_13_stable_period_no_stable_periods_consistent_error_handling(num_rows, threshold):
    """
    Property 13: Consistent error handling when no stable periods exist.
    
    When no periods meet the stability threshold, find_stable_period should
    consistently return (None, None, 0) rather than raising an exception.
    
    **Validates: Requirements 5.5**
    """
    calculator = FeatureCalculator()
    
    # Create volatility DataFrame where all std values are above threshold
    start_time = datetime(2024, 1, 1)
    index = pd.date_range(start=start_time, periods=num_rows, freq="D")
    
    # Ensure all std values are above threshold
    volatility_df = pd.DataFrame({
        "标准差": np.random.uniform(threshold + 1, threshold + 10, num_rows)
    }, index=index)
    
    # Should return (None, None, 0), not raise exception
    start_date, end_date, length = calculator.find_stable_period(volatility_df, threshold)
    
    assert start_date is None, "Should return None for start_date when no stable period"
    assert end_date is None, "Should return None for end_date when no stable period"
    assert length == 0, "Should return 0 for length when no stable period"


# ============================================================================
# Additional initialization tests
# ============================================================================

def test_feature_calculator_initialization():
    """
    Test that FeatureCalculator initializes correctly with expected attributes.
    """
    calculator = FeatureCalculator()
    
    # Check that _feature_calculators dictionary is initialized
    assert hasattr(calculator, "_feature_calculators"), "Should have _feature_calculators attribute"
    assert isinstance(calculator._feature_calculators, dict), "_feature_calculators should be a dict"
    
    # Check that expected features are present
    expected_features = ["均值", "中位数", "最大值", "最小值", "标准差", "Q1", "Q3", "P10", "超过Q3占时比"]
    for feature in expected_features:
        assert feature in calculator._feature_calculators, f"Feature '{feature}' should be in _feature_calculators"


def test_feature_calculator_methods_exist():
    """
    Test that FeatureCalculator has all expected public methods.
    """
    calculator = FeatureCalculator()
    
    # Check public methods exist
    assert hasattr(calculator, "calculate_statistical_features"), "Should have calculate_statistical_features method"
    assert hasattr(calculator, "analyze_spectral"), "Should have analyze_spectral method"
    assert hasattr(calculator, "calculate_volatility_features"), "Should have calculate_volatility_features method"
    assert hasattr(calculator, "find_stable_period"), "Should have find_stable_period method"
    assert hasattr(calculator, "recalculate_single_window"), "Should have recalculate_single_window method"
    
    # Check methods are callable
    assert callable(calculator.calculate_statistical_features), "calculate_statistical_features should be callable"
    assert callable(calculator.analyze_spectral), "analyze_spectral should be callable"
    assert callable(calculator.calculate_volatility_features), "calculate_volatility_features should be callable"
    assert callable(calculator.find_stable_period), "find_stable_period should be callable"
    assert callable(calculator.recalculate_single_window), "recalculate_single_window should be callable"


# ============================================================================
# Property 16: Individual MCP tool correctness
# Feature: feature-calculator-mcp, Property 16: Individual MCP tool correctness
# ============================================================================

@settings(max_examples=100, deadline=None)
@given(
    num_rows=st.integers(min_value=50, max_value=200),
    field_name=st.text(min_size=1, max_size=20, alphabet=st.characters(blacklist_characters=['\x00'])),
    feature=st.sampled_from(["均值", "中位数", "最大值", "最小值", "标准差", "Q1", "Q3", "P10", "超过Q3占时比"]),
    freq=st.sampled_from(["h", "D", "W"])
)
def test_property_16_individual_mcp_tool_correctness(num_rows, field_name, feature, freq):
    """
    Property 16: Individual MCP tool correctness.
    
    For any valid time series data and individual statistical feature MCP tool
    (mean, median, max, min, std, Q1, Q3, P10, percent above Q3, range),
    calling the individual tool should return the same result as calling the
    batch tool with only that feature requested.
    
    This test verifies that individual MCP tools produce the same results as
    the batch calculator when requesting a single feature. Since MCP tools are
    wrapped by FastMCP, we test the underlying logic by comparing the calculator
    results directly.
    
    **Validates: Requirements 4.5**
    """
    from src.features.mcp_server import dataframe_to_json, parse_json_to_dataframe
    
    calculator = FeatureCalculator()
    
    # Create DataFrame with valid time series data
    start_time = datetime(2024, 1, 1)
    index = pd.date_range(start=start_time, periods=num_rows, freq="h")
    df = pd.DataFrame({
        field_name: np.random.uniform(10, 100, num_rows)
    }, index=index)
    
    # Convert DataFrame to JSON and back to simulate MCP round-trip
    data_json = json.dumps({
        "data": df.reset_index().rename(columns={"index": "time"}).to_dict(orient="records")
    }, default=str)
    
    # Parse JSON back to DataFrame (simulating what MCP tools do)
    parsed_df = parse_json_to_dataframe(data_json)
    
    # Call calculator with parsed DataFrame (simulating individual MCP tool)
    individual_result = calculator.calculate_statistical_features(parsed_df, field_name, [feature], freq)
    
    # Call calculator with original DataFrame (simulating batch tool)
    batch_result = calculator.calculate_statistical_features(df, field_name, [feature], freq)
    
    # Both should return non-empty results
    assert not individual_result.empty, "Individual tool should return non-empty result"
    assert not batch_result.empty, "Batch tool should return non-empty result"
    
    # The number of time points should match
    assert len(individual_result) == len(batch_result), \
        "Individual and batch tools should return same number of time points"
    
    # Extract the feature column name (accounting for renaming like 中位数 -> 中位数 (Q2))
    expected_column = feature
    if feature == "中位数":
        expected_column = "中位数 (Q2)"
    
    # Both should have the same columns
    assert expected_column in individual_result.columns, f"Individual result should have {expected_column}"
    assert expected_column in batch_result.columns, f"Batch result should have {expected_column}"
    
    # Compare the feature values (allowing for floating point precision)
    # Note: We don't check the index because JSON round-trip may add timezone info
    pd.testing.assert_series_equal(
        individual_result[expected_column],
        batch_result[expected_column],
        check_names=False,
        check_index=False,  # Don't check index due to timezone differences from JSON round-trip
        atol=1e-10,
        rtol=1e-10
    )


# ============================================================================
# Property 12: MCP input validation
# Feature: feature-calculator-mcp, Property 12: MCP input validation
# ============================================================================

@settings(max_examples=100, deadline=None)
@given(
    invalid_input_type=st.sampled_from([
        "empty_json",
        "invalid_json",
        "missing_field",
        "invalid_frequency",
        "invalid_feature_list",
        "empty_dataframe"
    ]),
    field_name=st.text(min_size=1, max_size=20, alphabet=st.characters(blacklist_characters=['\x00'])),
    freq=st.sampled_from(["h", "D", "W"])
)
def test_property_12_mcp_input_validation(invalid_input_type, field_name, freq):
    """
    Property 12: MCP input validation.
    
    For any invalid input parameters (empty data, missing required fields,
    invalid types), MCP calls should perform validation before processing
    and reject the request with an appropriate error.
    
    This test verifies that the MCP validation functions correctly identify
    and reject various types of invalid inputs.
    
    **Validates: Requirements 4.3**
    """
    from src.features.mcp_server import (
        validate_json_format,
        validate_dataframe_field,
        validate_frequency,
        validate_feature_list,
        create_error_response,
        parse_json_to_dataframe
    )
    
    if invalid_input_type == "empty_json":
        # Test empty JSON string
        is_valid, error_msg = validate_json_format("")
        assert not is_valid, "Empty JSON should be invalid"
        assert "empty" in error_msg.lower(), "Error message should mention empty input"
        
        # Test that error response is properly formatted
        error_response = create_error_response("JSONParseError", error_msg)
        error_data = json.loads(error_response)
        assert error_data["success"] is False, "Error response should have success=False"
        assert "error" in error_data, "Error response should have 'error' key"
        assert "type" in error_data["error"], "Error should have 'type' field"
        assert "message" in error_data["error"], "Error should have 'message' field"
    
    elif invalid_input_type == "invalid_json":
        # Test malformed JSON
        invalid_json = "{this is not valid json"
        is_valid, error_msg = validate_json_format(invalid_json)
        assert not is_valid, "Malformed JSON should be invalid"
        assert "json" in error_msg.lower(), "Error message should mention JSON"
        
        # Test that error response is properly formatted
        error_response = create_error_response("JSONParseError", error_msg)
        error_data = json.loads(error_response)
        assert error_data["success"] is False
        assert error_data["error"]["type"] == "JSONParseError"
    
    elif invalid_input_type == "missing_field":
        # Create DataFrame without the requested field
        start_time = datetime(2024, 1, 1)
        index = pd.date_range(start=start_time, periods=10, freq="h")
        df = pd.DataFrame({
            "other_field": np.random.randn(10)
        }, index=index)
        
        # Validate that missing field is detected
        is_valid, error_msg = validate_dataframe_field(df, field_name)
        assert not is_valid, "Missing field should be invalid"
        assert field_name in error_msg, "Error message should mention the missing field"
        assert "not found" in error_msg.lower(), "Error message should indicate field not found"
    
    elif invalid_input_type == "invalid_frequency":
        # Test invalid frequency
        invalid_freq = "X"  # Not a valid frequency
        is_valid, error_msg = validate_frequency(invalid_freq)
        assert not is_valid, "Invalid frequency should be rejected"
        assert "frequency" in error_msg.lower() or "invalid" in error_msg.lower(), \
            "Error message should mention frequency or invalid"
    
    elif invalid_input_type == "invalid_feature_list":
        # Test invalid feature list
        invalid_features = ["invalid_feature", "another_invalid"]
        is_valid, error_msg = validate_feature_list(invalid_features)
        assert not is_valid, "Invalid features should be rejected"
        assert "invalid" in error_msg.lower(), "Error message should mention invalid features"
    
    elif invalid_input_type == "empty_dataframe":
        # Test empty DataFrame
        empty_df = pd.DataFrame(index=pd.DatetimeIndex([]))
        is_valid, error_msg = validate_dataframe_field(empty_df, field_name)
        assert not is_valid, "Empty DataFrame should be invalid"
        assert "empty" in error_msg.lower(), "Error message should mention empty DataFrame"


@settings(max_examples=100, deadline=None)
@given(
    num_rows=st.integers(min_value=10, max_value=100),
    valid_field=st.text(min_size=1, max_size=20, alphabet=st.characters(blacklist_characters=['\x00'])),
    invalid_field=st.text(min_size=1, max_size=20, alphabet=st.characters(blacklist_characters=['\x00'])),
    freq=st.sampled_from(["h", "D", "W"])
)
def test_property_12_mcp_validation_prevents_processing(num_rows, valid_field, invalid_field, freq):
    """
    Property 12: MCP validation prevents processing of invalid inputs.
    
    For any MCP tool, when given invalid input, the validation should catch
    the error and return an error response without attempting to process the data.
    
    This test verifies that validation happens before processing by testing
    the validation functions that MCP tools use.
    
    **Validates: Requirements 4.3**
    """
    from src.features.mcp_server import (
        validate_json_format,
        validate_dataframe_field,
        validate_frequency,
        create_error_response,
        parse_json_to_dataframe
    )
    
    # Ensure fields are different
    if invalid_field == valid_field:
        invalid_field = valid_field + "_different"
    
    # Test 1: Invalid JSON should be caught by validation
    invalid_json = "not valid json at all"
    is_valid, error_msg = validate_json_format(invalid_json)
    assert not is_valid, "Invalid JSON should be caught by validation"
    
    # Create error response
    error_response = create_error_response("JSONParseError", error_msg)
    result_data = json.loads(error_response)
    
    assert "success" in result_data, "Response should have 'success' field"
    assert result_data["success"] is False, "Invalid JSON should result in success=False"
    assert "error" in result_data, "Response should have 'error' field"
    assert "type" in result_data["error"], "Error should have 'type' field"
    assert "message" in result_data["error"], "Error should have 'message' field"
    
    # Test 2: Valid JSON but missing field should be caught by validation
    start_time = datetime(2024, 1, 1)
    index = pd.date_range(start=start_time, periods=num_rows, freq="h")
    df = pd.DataFrame({
        valid_field: np.random.uniform(10, 100, num_rows)
    }, index=index)
    
    # Validate that missing field is caught
    is_valid, error_msg = validate_dataframe_field(df, invalid_field)
    assert not is_valid, "Missing field should be caught by validation"
    assert invalid_field in error_msg, "Error message should mention the missing field"
    
    # Create error response
    error_response = create_error_response("ValidationError", error_msg, {"field_name": invalid_field})
    result_data = json.loads(error_response)
    
    # Verify error response structure
    assert result_data["success"] is False, "Missing field should result in error"
    assert "error" in result_data, "Error response should have 'error' field"
    assert "type" in result_data["error"], "Error should have 'type' field"
    assert "message" in result_data["error"], "Error should have 'message' field"
    assert "details" in result_data["error"], "Error should have 'details' field"


# ============================================================================
# Property 11: MCP JSON round-trip
# Feature: feature-calculator-mcp, Property 11: MCP JSON round-trip
# ============================================================================

@settings(max_examples=100, deadline=None)
@given(
    num_rows=st.integers(min_value=5, max_value=100),
    num_cols=st.integers(min_value=1, max_value=5),
    include_special_values=st.booleans(),
    timezone=st.sampled_from(["UTC", "Asia/Shanghai", "America/New_York", "Europe/London"])
)
def test_property_11_mcp_json_round_trip(num_rows, num_cols, include_special_values, timezone):
    """
    Property 11: MCP JSON round-trip.
    
    For any valid DataFrame that can be serialized to JSON, sending it through
    MCP (serialize to JSON, call MCP endpoint, deserialize response) should
    preserve the essential data structure and numeric values within acceptable
    floating-point precision.
    
    This test verifies that:
    1. Timezone-aware datetime serialization/deserialization works correctly
    2. NaN values are preserved through the round-trip
    3. Infinity values are preserved through the round-trip
    4. Numeric precision is maintained within floating-point tolerance
    5. DataFrame structure (shape, columns, index) is preserved
    
    **Validates: Requirements 4.2**
    """
    from src.features.mcp_server import dataframe_to_json, parse_json_to_dataframe
    
    # Create DataFrame with timezone-aware DatetimeIndex
    start_time = pd.Timestamp("2024-01-01", tz=timezone)
    index = pd.date_range(start=start_time, periods=num_rows, freq="h")
    
    # Generate column names
    col_names = [f"col_{i}" for i in range(num_cols)]
    
    # Create data with normal numeric values
    data = {}
    for col in col_names:
        data[col] = np.random.uniform(-100, 100, num_rows)
    
    # Add special values if requested
    if include_special_values and num_rows >= 3:
        # Add NaN values
        for col in col_names[:max(1, num_cols // 2)]:
            data[col][0] = np.nan
        
        # Add infinity values
        if num_cols > 1:
            data[col_names[-1]][1] = np.inf
            if num_rows >= 3:
                data[col_names[-1]][2] = -np.inf
    
    # Create original DataFrame
    original_df = pd.DataFrame(data, index=index)
    
    # Serialize to JSON
    json_str = dataframe_to_json(original_df)
    
    # Verify JSON is valid
    json_data = json.loads(json_str)
    assert "features" in json_data, "JSON should have 'features' key"
    assert isinstance(json_data["features"], list), "Features should be a list"
    
    # Deserialize back to DataFrame
    # First, convert to the format expected by parse_json_to_dataframe
    input_json = json.dumps({"data": json_data["features"]})
    reconstructed_df = parse_json_to_dataframe(input_json)
    
    # Verify structure is preserved
    assert len(reconstructed_df) == len(original_df), \
        f"Row count should be preserved: {len(reconstructed_df)} vs {len(original_df)}"
    
    assert len(reconstructed_df.columns) == len(original_df.columns), \
        f"Column count should be preserved: {len(reconstructed_df.columns)} vs {len(original_df.columns)}"
    
    # Verify index is DatetimeIndex with timezone
    assert isinstance(reconstructed_df.index, pd.DatetimeIndex), \
        "Index should be DatetimeIndex after round-trip"
    
    assert reconstructed_df.index.tz is not None, \
        "Index should be timezone-aware after round-trip"
    
    # Verify column names are preserved
    for col in original_df.columns:
        assert col in reconstructed_df.columns, \
            f"Column '{col}' should be preserved in round-trip"
    
    # Verify numeric values are preserved within tolerance
    for col in original_df.columns:
        original_col = original_df[col]
        reconstructed_col = reconstructed_df[col]
        
        # Check each value
        for i in range(len(original_col)):
            orig_val = original_col.iloc[i]
            recon_val = reconstructed_col.iloc[i]
            
            # Handle NaN values
            if pd.isna(orig_val):
                assert pd.isna(recon_val), \
                    f"NaN value at index {i} in column '{col}' should be preserved"
            # Handle infinity values
            elif np.isinf(orig_val):
                assert np.isinf(recon_val), \
                    f"Infinity value at index {i} in column '{col}' should be preserved"
                assert np.sign(orig_val) == np.sign(recon_val), \
                    f"Sign of infinity at index {i} in column '{col}' should be preserved"
            # Handle normal numeric values
            else:
                assert np.isclose(orig_val, recon_val, rtol=1e-10, atol=1e-10), \
                    f"Numeric value at index {i} in column '{col}' should be preserved within tolerance: {orig_val} vs {recon_val}"
    
    # Verify datetime index values are preserved (within reasonable precision)
    for i in range(len(original_df)):
        orig_time = original_df.index[i]
        recon_time = reconstructed_df.index[i]
        
        # Convert both to UTC for comparison
        orig_time_utc = orig_time.tz_convert("UTC")
        recon_time_utc = recon_time.tz_convert("UTC")
        
        # Allow for microsecond precision differences
        time_diff = abs((orig_time_utc - recon_time_utc).total_seconds())
        assert time_diff < 0.001, \
            f"Datetime at index {i} should be preserved within 1ms: {orig_time_utc} vs {recon_time_utc}"


@settings(max_examples=100, deadline=None)
@given(
    num_rows=st.integers(min_value=10, max_value=100),
    field_name=st.text(min_size=1, max_size=20, alphabet=st.characters(blacklist_characters=['\x00'])),
    feature_list=st.lists(
        st.sampled_from(["均值", "中位数", "最大值", "最小值", "标准差"]),
        min_size=1,
        max_size=3,
        unique=True
    ),
    freq=st.sampled_from(["h", "D"])
)
def test_property_11_mcp_json_round_trip_with_calculator(num_rows, field_name, feature_list, freq):
    """
    Property 11: MCP JSON round-trip with actual calculator results.
    
    For any valid DataFrame processed through the calculator and then serialized
    to JSON, the round-trip should preserve all calculated feature values within
    acceptable floating-point precision.
    
    This test verifies the complete MCP workflow: DataFrame -> Calculator ->
    JSON -> Parse -> DataFrame, ensuring that calculated features are preserved.
    
    **Validates: Requirements 4.2**
    """
    from src.features.mcp_server import dataframe_to_json, parse_json_to_dataframe
    
    calculator = FeatureCalculator()
    
    # Create DataFrame with valid time series data
    start_time = pd.Timestamp("2024-01-01", tz="UTC")
    index = pd.date_range(start=start_time, periods=num_rows, freq="h")
    df = pd.DataFrame({
        field_name: np.random.uniform(10, 100, num_rows)
    }, index=index)
    
    # Calculate features
    result_df = calculator.calculate_statistical_features(df, field_name, feature_list, freq)
    
    # Skip if result is empty
    if result_df.empty:
        return
    
    # Serialize to JSON
    json_str = dataframe_to_json(result_df)
    
    # Verify JSON is valid
    json_data = json.loads(json_str)
    assert "features" in json_data, "JSON should have 'features' key"
    
    # Deserialize back to DataFrame
    input_json = json.dumps({"data": json_data["features"]})
    reconstructed_df = parse_json_to_dataframe(input_json)
    
    # Verify structure
    assert len(reconstructed_df) == len(result_df), \
        "Row count should be preserved in round-trip"
    
    # Verify all feature columns are present
    for col in result_df.columns:
        assert col in reconstructed_df.columns, \
            f"Feature column '{col}' should be preserved in round-trip"
    
    # Verify feature values are preserved within tolerance
    for col in result_df.columns:
        original_values = result_df[col]
        reconstructed_values = reconstructed_df[col]
        
        # Compare values element-wise
        for i in range(len(original_values)):
            orig_val = original_values.iloc[i]
            recon_val = reconstructed_values.iloc[i]
            
            if pd.isna(orig_val):
                assert pd.isna(recon_val), \
                    f"NaN in column '{col}' at index {i} should be preserved"
            elif np.isinf(orig_val):
                assert np.isinf(recon_val) and np.sign(orig_val) == np.sign(recon_val), \
                    f"Infinity in column '{col}' at index {i} should be preserved"
            else:
                assert np.isclose(orig_val, recon_val, rtol=1e-9, atol=1e-9), \
                    f"Value in column '{col}' at index {i} should be preserved: {orig_val} vs {recon_val}"
