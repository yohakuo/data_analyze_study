"""
Features Module - Time Series Feature Calculation

This module provides a comprehensive suite of feature calculation capabilities
for time series data analysis. It includes statistical aggregations, spectral
analysis using Fast Fourier Transform (FFT), and volatility metrics.

Main Components
---------------
FeatureCalculator : class
    The primary interface for calculating features from time series data.
    Provides methods for statistical features, spectral analysis, and
    volatility calculations.

mcp_server : module
    FastMCP server that exposes FeatureCalculator functionality through
    the Model Context Protocol (MCP), enabling external systems to invoke
    feature calculations via standardized API calls.

Feature Types
-------------
Statistical Features:
    - Mean (均值)
    - Median (中位数)
    - Maximum (最大值)
    - Minimum (最小值)
    - Standard Deviation (标准差)
    - Quartiles (Q1, Q3)
    - Percentiles (P10)
    - Percent Above Q3 (超过Q3占时比)
    - Range (极差) - automatically calculated when max and min are requested
    - Range Rate of Change (极差的时间变化率)

Spectral Features:
    - FFT-based periodic pattern identification
    - Detrended spectral analysis
    - Period and amplitude extraction

Volatility Features:
    - Daily mean and standard deviation
    - Mean Absolute Deviation (MAD) from mean and median
    - First-order autocorrelation
    - Coefficient of variation
    - Stable period identification

Usage Examples
--------------
Basic statistical feature calculation:

    >>> from src.features import FeatureCalculator
    >>> import pandas as pd
    >>> 
    >>> # Create calculator instance
    >>> calculator = FeatureCalculator()
    >>> 
    >>> # Load time series data with DatetimeIndex
    >>> df = pd.read_csv("data.csv", index_col="time", parse_dates=True)
    >>> 
    >>> # Calculate statistical features
    >>> features = calculator.calculate_statistical_features(
    ...     df=df,
    ...     field_name="temperature",
    ...     feature_list=["均值", "最大值", "最小值", "标准差"],
    ...     freq="D"  # Daily aggregation
    ... )
    >>> print(features)

Spectral analysis:

    >>> # Identify periodic patterns
    >>> spectrum = calculator.analyze_spectral(df, "temperature")
    >>> 
    >>> # Find dominant period
    >>> dominant_period = spectrum.loc[
    ...     spectrum["强度(幅度)"].idxmax(), 
    ...     "周期(小时)"
    ... ]
    >>> print(f"Dominant period: {dominant_period} hours")

Volatility analysis:

    >>> # Calculate daily volatility metrics
    >>> volatility = calculator.calculate_volatility_features(df, "temperature")
    >>> 
    >>> # Find stable periods
    >>> start, end, length = calculator.find_stable_period(
    ...     volatility, 
    ...     threshold=1.5
    ... )
    >>> print(f"Stable period: {start} to {end} ({length} days)")

MCP Server Usage:

    The MCP server can be started to expose feature calculation through
    the Model Context Protocol. See src/features/mcp_server.py for details
    on available MCP tools and their parameters.

Notes
-----
- All time series data must use pandas DataFrames with DatetimeIndex
- Timezone-aware datetime indices are supported
- NaN values are handled gracefully in calculations
- Logging is configured for all operations with structured context

See Also
--------
src.features.calculator : FeatureCalculator class implementation
src.features.mcp_server : MCP server implementation
"""

from src.features.calculator import FeatureCalculator

__all__ = ["FeatureCalculator"]

__version__ = "1.0.0"
__author__ = "Feature Calculator Team"
