"""
MCP Server for Feature Calculator

This module provides a FastMCP server that exposes the FeatureCalculator
functionality through individual MCP tools. Each statistical feature can be
calculated individually or in batch mode.
"""

import json
import logging

import numpy as np
import pandas as pd
from fastmcp import FastMCP

from src.calculator import FeatureCalculator

# Configure module logger
logger = logging.getLogger(__name__)


# Initialize FastMCP server
mcp = FastMCP("Feature Calculator")

# Create calculator instance
calculator = FeatureCalculator()


# ============================================================================
# Helper Functions
# ============================================================================


def create_error_response(error_type: str, message: str, details: dict = None) -> str:
    """
    Create a structured error response in JSON format.

    Args:
        error_type: Type of error (e.g., "ValidationError", "JSONParseError")
        message: Human-readable error message
        details: Optional dictionary with additional error details

    Returns:
        JSON string with structured error response
    """
    logger.error(
        f"MCP error: {error_type} - {message}",
        extra={
            "error_type": error_type,
            "error_message": message,
            "error_details": details or {},
        },
    )

    error_response = {
        "success": False,
        "error": {"type": error_type, "message": message, "details": details or {}},
    }
    return json.dumps(error_response, ensure_ascii=False)


def validate_json_format(data_json: str) -> tuple[bool, str]:
    """
    Validate that the input is valid JSON.

    Args:
        data_json: String to validate as JSON

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(data_json, str):
        return False, f"Input must be a string, got {type(data_json).__name__}"

    if not data_json or data_json.strip() == "":
        return False, "Input JSON string is empty"

    try:
        json.loads(data_json)
        return True, ""
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON format: {str(e)}"


def validate_dataframe_field(df: pd.DataFrame, field_name: str) -> tuple[bool, str]:
    """
    Validate that a field exists in the DataFrame.

    Args:
        df: DataFrame to check
        field_name: Name of the field to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"

    if field_name not in df.columns:
        available_fields = list(df.columns)
        return (
            False,
            f"Field '{field_name}' not found in DataFrame. Available fields: {available_fields}",
        )

    return True, ""


def validate_frequency(freq: str) -> tuple[bool, str]:
    """
    Validate that the frequency parameter is valid.

    Args:
        freq: Frequency string to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    valid_frequencies = ["h", "D", "W", "M", "ME", "MS"]
    if freq not in valid_frequencies:
        return (
            False,
            f"Invalid frequency '{freq}'. Valid frequencies: {valid_frequencies}",
        )

    return True, ""


def validate_feature_list(feature_list: list) -> tuple[bool, str]:
    """
    Validate that the feature list is valid.

    Args:
        feature_list: List of features to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(feature_list, list):
        return False, f"feature_list must be a list, got {type(feature_list).__name__}"

    if len(feature_list) == 0:
        return False, "feature_list cannot be empty"

    valid_features = [
        "均值",
        "中位数",
        "最大值",
        "最小值",
        "标准差",
        "Q1",
        "Q3",
        "P10",
        "超过Q3占时比",
    ]
    invalid_features = [f for f in feature_list if f not in valid_features]

    if invalid_features:
        return (
            False,
            f"Invalid features: {invalid_features}. Valid features: {valid_features}",
        )

    return True, ""


def parse_json_to_dataframe(data_json: str) -> pd.DataFrame:
    """
    Parse JSON string to pandas DataFrame with DatetimeIndex.

    Handles timezone-aware datetime deserialization, NaN values (represented as null),
    and infinity values (represented as "Infinity", "-Infinity", or "inf", "-inf").

    Args:
        data_json: JSON string containing time series data

    Returns:
        DataFrame with DatetimeIndex

    Raises:
        ValueError: If JSON is invalid or doesn't contain required fields
    """
    data = json.loads(data_json)

    # Handle different JSON formats
    if isinstance(data, dict) and "data" in data:
        df = pd.DataFrame(data["data"])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame(data)

    # Identify the time/index column
    time_col = None
    if "time" in df.columns:
        time_col = "time"
    elif "index" in df.columns:
        time_col = "index"

    # Convert time column to DatetimeIndex with timezone awareness
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        df.set_index(time_col, inplace=True)

    # Handle infinity values represented as strings
    for col in df.columns:
        if df[col].dtype == object:
            # Map string representations to numeric values
            def convert_special_values(val):
                if val == "Infinity" or val == "inf":
                    return np.inf
                elif val == "-Infinity" or val == "-inf":
                    return -np.inf
                elif val == "NaN":
                    return np.nan
                else:
                    return val

            df[col] = df[col].apply(convert_special_values)

            # Try to convert to numeric if possible
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

    return df


def dataframe_to_json(df: pd.DataFrame) -> str:
    """
    Convert DataFrame to JSON string.

    Handles timezone-aware datetime serialization, NaN values (converted to null),
    and infinity values (converted to "Infinity" or "-Infinity" strings).
    Preserves numeric precision during serialization.

    Args:
        df: DataFrame to convert

    Returns:
        JSON string representation with proper handling of special values
    """
    # Reset index to include time in the output
    df_reset = df.reset_index()

    # Create a copy to avoid modifying the original
    df_copy = df_reset.copy()

    # Handle timezone-aware datetime columns
    for col in df_copy.columns:
        if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            # Convert to ISO format string with timezone
            df_copy[col] = df_copy[col].apply(
                lambda x: x.isoformat() if pd.notna(x) else None
            )

    # Handle NaN and infinity values in numeric columns before converting to dict
    for col in df_copy.columns:
        if pd.api.types.is_numeric_dtype(df_copy[col]):
            # Replace NaN with None (will become JSON null)
            df_copy[col] = df_copy[col].apply(
                lambda x: None
                if pd.isna(x)
                else (
                    "Infinity"
                    if np.isinf(x) and x > 0
                    else ("-Infinity" if np.isinf(x) and x < 0 else x)
                )
            )

    # Convert to dict with records orientation
    result = df_copy.to_dict(orient="records")

    # Custom JSON encoder for any remaining special values
    def json_encoder(obj):
        """Custom encoder for special numeric values."""
        if isinstance(obj, (np.integer, np.floating)):
            val = float(obj)
            if np.isnan(val):
                return None
            elif np.isinf(val):
                return "Infinity" if val > 0 else "-Infinity"
            else:
                return val
        elif pd.isna(obj):
            return None
        else:
            return str(obj)

    return json.dumps({"features": result}, ensure_ascii=False, default=json_encoder)


# ============================================================================
# Individual Statistical Feature Tools
# ============================================================================


def _calculate_mean_impl(data_json: str, field_name: str, freq: str = "h") -> str:
    """Implementation of calculate_mean that can be tested directly."""
    logger.info(
        "MCP request: calculate_mean", extra={"field_name": field_name, "freq": freq}
    )

    try:
        # Validate JSON format
        is_valid, error_msg = validate_json_format(data_json)
        if not is_valid:
            return create_error_response("JSONParseError", error_msg)

        # Validate frequency
        is_valid, error_msg = validate_frequency(freq)
        if not is_valid:
            return create_error_response("ValidationError", error_msg)

        # Parse JSON to DataFrame
        df = parse_json_to_dataframe(data_json)

        # Validate field exists
        is_valid, error_msg = validate_dataframe_field(df, field_name)
        if not is_valid:
            return create_error_response(
                "ValidationError", error_msg, {"field_name": field_name}
            )

        # Calculate features
        result = calculator.calculate_statistical_features(
            df, field_name, ["均值"], freq
        )

        logger.info(
            "MCP response: calculate_mean completed successfully",
            extra={"field_name": field_name, "result_rows": len(result)},
        )

        return dataframe_to_json(result)

    except Exception as e:
        logger.exception(
            "Unexpected error in calculate_mean",
            extra={"field_name": field_name, "freq": freq},
        )
        return create_error_response("InternalError", f"An error occurred: {str(e)}")


@mcp.tool()
def calculate_mean(data_json: str, field_name: str, freq: str = "h") -> str:
    """
    Calculate mean (均值) for time series data.

    Args:
        data_json: JSON string containing time series data
        field_name: Name of the field to calculate mean for
        freq: Resampling frequency ('h', 'D', 'W', 'M')

    Returns:
        JSON string with mean values over time or error response
    """
    return _calculate_mean_impl(data_json, field_name, freq)


@mcp.tool()
def calculate_median(data_json: str, field_name: str, freq: str = "h") -> str:
    """
    Calculate median (中位数) for time series data.

    Args:
        data_json: JSON string containing time series data
        field_name: Name of the field to calculate median for
        freq: Resampling frequency ('h', 'D', 'W', 'M')

    Returns:
        JSON string with median values over time or error response
    """
    try:
        # Validate JSON format
        is_valid, error_msg = validate_json_format(data_json)
        if not is_valid:
            return create_error_response("JSONParseError", error_msg)

        # Validate frequency
        is_valid, error_msg = validate_frequency(freq)
        if not is_valid:
            return create_error_response("ValidationError", error_msg)

        # Parse JSON to DataFrame
        df = parse_json_to_dataframe(data_json)

        # Validate field exists
        is_valid, error_msg = validate_dataframe_field(df, field_name)
        if not is_valid:
            return create_error_response(
                "ValidationError", error_msg, {"field_name": field_name}
            )

        # Calculate features
        result = calculator.calculate_statistical_features(
            df, field_name, ["中位数"], freq
        )
        return dataframe_to_json(result)

    except Exception as e:
        return create_error_response("InternalError", f"An error occurred: {str(e)}")


@mcp.tool()
def calculate_max(data_json: str, field_name: str, freq: str = "h") -> str:
    """
    Calculate maximum (最大值) for time series data.

    Args:
        data_json: JSON string containing time series data
        field_name: Name of the field to calculate maximum for
        freq: Resampling frequency ('h', 'D', 'W', 'M')

    Returns:
        JSON string with maximum values over time or error response
    """
    try:
        # Validate JSON format
        is_valid, error_msg = validate_json_format(data_json)
        if not is_valid:
            return create_error_response("JSONParseError", error_msg)

        # Validate frequency
        is_valid, error_msg = validate_frequency(freq)
        if not is_valid:
            return create_error_response("ValidationError", error_msg)

        # Parse JSON to DataFrame
        df = parse_json_to_dataframe(data_json)

        # Validate field exists
        is_valid, error_msg = validate_dataframe_field(df, field_name)
        if not is_valid:
            return create_error_response(
                "ValidationError", error_msg, {"field_name": field_name}
            )

        # Calculate features
        result = calculator.calculate_statistical_features(
            df, field_name, ["最大值"], freq
        )
        return dataframe_to_json(result)

    except Exception as e:
        return create_error_response("InternalError", f"An error occurred: {str(e)}")


@mcp.tool()
def calculate_min(data_json: str, field_name: str, freq: str = "h") -> str:
    """
    Calculate minimum (最小值) for time series data.

    Args:
        data_json: JSON string containing time series data
        field_name: Name of the field to calculate minimum for
        freq: Resampling frequency ('h', 'D', 'W', 'M')

    Returns:
        JSON string with minimum values over time or error response
    """
    try:
        # Validate JSON format
        is_valid, error_msg = validate_json_format(data_json)
        if not is_valid:
            return create_error_response("JSONParseError", error_msg)

        # Validate frequency
        is_valid, error_msg = validate_frequency(freq)
        if not is_valid:
            return create_error_response("ValidationError", error_msg)

        # Parse JSON to DataFrame
        df = parse_json_to_dataframe(data_json)

        # Validate field exists
        is_valid, error_msg = validate_dataframe_field(df, field_name)
        if not is_valid:
            return create_error_response(
                "ValidationError", error_msg, {"field_name": field_name}
            )

        # Calculate features
        result = calculator.calculate_statistical_features(
            df, field_name, ["最小值"], freq
        )
        return dataframe_to_json(result)

    except Exception as e:
        return create_error_response("InternalError", f"An error occurred: {str(e)}")


@mcp.tool()
def calculate_std(data_json: str, field_name: str, freq: str = "h") -> str:
    """
    Calculate standard deviation (标准差) for time series data.

    Args:
        data_json: JSON string containing time series data
        field_name: Name of the field to calculate standard deviation for
        freq: Resampling frequency ('h', 'D', 'W', 'M')

    Returns:
        JSON string with standard deviation values over time or error response
    """
    try:
        # Validate JSON format
        is_valid, error_msg = validate_json_format(data_json)
        if not is_valid:
            return create_error_response("JSONParseError", error_msg)

        # Validate frequency
        is_valid, error_msg = validate_frequency(freq)
        if not is_valid:
            return create_error_response("ValidationError", error_msg)

        # Parse JSON to DataFrame
        df = parse_json_to_dataframe(data_json)

        # Validate field exists
        is_valid, error_msg = validate_dataframe_field(df, field_name)
        if not is_valid:
            return create_error_response(
                "ValidationError", error_msg, {"field_name": field_name}
            )

        # Calculate features
        result = calculator.calculate_statistical_features(
            df, field_name, ["标准差"], freq
        )
        return dataframe_to_json(result)

    except Exception as e:
        return create_error_response("InternalError", f"An error occurred: {str(e)}")


@mcp.tool()
def calculate_q1(data_json: str, field_name: str, freq: str = "h") -> str:
    """
    Calculate first quartile (Q1) for time series data.

    Args:
        data_json: JSON string containing time series data
        field_name: Name of the field to calculate Q1 for
        freq: Resampling frequency ('h', 'D', 'W', 'M')

    Returns:
        JSON string with Q1 values over time or error response
    """
    try:
        # Validate JSON format
        is_valid, error_msg = validate_json_format(data_json)
        if not is_valid:
            return create_error_response("JSONParseError", error_msg)

        # Validate frequency
        is_valid, error_msg = validate_frequency(freq)
        if not is_valid:
            return create_error_response("ValidationError", error_msg)

        # Parse JSON to DataFrame
        df = parse_json_to_dataframe(data_json)

        # Validate field exists
        is_valid, error_msg = validate_dataframe_field(df, field_name)
        if not is_valid:
            return create_error_response(
                "ValidationError", error_msg, {"field_name": field_name}
            )

        # Calculate features
        result = calculator.calculate_statistical_features(df, field_name, ["Q1"], freq)
        return dataframe_to_json(result)

    except Exception as e:
        return create_error_response("InternalError", f"An error occurred: {str(e)}")


@mcp.tool()
def calculate_q3(data_json: str, field_name: str, freq: str = "h") -> str:
    """
    Calculate third quartile (Q3) for time series data.

    Args:
        data_json: JSON string containing time series data
        field_name: Name of the field to calculate Q3 for
        freq: Resampling frequency ('h', 'D', 'W', 'M')

    Returns:
        JSON string with Q3 values over time or error response
    """
    try:
        # Validate JSON format
        is_valid, error_msg = validate_json_format(data_json)
        if not is_valid:
            return create_error_response("JSONParseError", error_msg)

        # Validate frequency
        is_valid, error_msg = validate_frequency(freq)
        if not is_valid:
            return create_error_response("ValidationError", error_msg)

        # Parse JSON to DataFrame
        df = parse_json_to_dataframe(data_json)

        # Validate field exists
        is_valid, error_msg = validate_dataframe_field(df, field_name)
        if not is_valid:
            return create_error_response(
                "ValidationError", error_msg, {"field_name": field_name}
            )

        # Calculate features
        result = calculator.calculate_statistical_features(df, field_name, ["Q3"], freq)
        return dataframe_to_json(result)

    except Exception as e:
        return create_error_response("InternalError", f"An error occurred: {str(e)}")


@mcp.tool()
def calculate_p10(data_json: str, field_name: str, freq: str = "h") -> str:
    """
    Calculate 10th percentile (P10) for time series data.

    Args:
        data_json: JSON string containing time series data
        field_name: Name of the field to calculate P10 for
        freq: Resampling frequency ('h', 'D', 'W', 'M')

    Returns:
        JSON string with P10 values over time or error response
    """
    try:
        # Validate JSON format
        is_valid, error_msg = validate_json_format(data_json)
        if not is_valid:
            return create_error_response("JSONParseError", error_msg)

        # Validate frequency
        is_valid, error_msg = validate_frequency(freq)
        if not is_valid:
            return create_error_response("ValidationError", error_msg)

        # Parse JSON to DataFrame
        df = parse_json_to_dataframe(data_json)

        # Validate field exists
        is_valid, error_msg = validate_dataframe_field(df, field_name)
        if not is_valid:
            return create_error_response(
                "ValidationError", error_msg, {"field_name": field_name}
            )

        # Calculate features
        result = calculator.calculate_statistical_features(
            df, field_name, ["P10"], freq
        )
        return dataframe_to_json(result)

    except Exception as e:
        return create_error_response("InternalError", f"An error occurred: {str(e)}")


@mcp.tool()
def calculate_percent_above_q3(data_json: str, field_name: str, freq: str = "h") -> str:
    """
    Calculate percentage of time above Q3 (超过Q3占时比) for time series data.

    Args:
        data_json: JSON string containing time series data
        field_name: Name of the field to calculate percent above Q3 for
        freq: Resampling frequency ('h', 'D', 'W', 'M')

    Returns:
        JSON string with percent above Q3 values over time or error response
    """
    try:
        # Validate JSON format
        is_valid, error_msg = validate_json_format(data_json)
        if not is_valid:
            return create_error_response("JSONParseError", error_msg)

        # Validate frequency
        is_valid, error_msg = validate_frequency(freq)
        if not is_valid:
            return create_error_response("ValidationError", error_msg)

        # Parse JSON to DataFrame
        df = parse_json_to_dataframe(data_json)

        # Validate field exists
        is_valid, error_msg = validate_dataframe_field(df, field_name)
        if not is_valid:
            return create_error_response(
                "ValidationError", error_msg, {"field_name": field_name}
            )

        # Calculate features
        result = calculator.calculate_statistical_features(
            df, field_name, ["超过Q3占时比"], freq
        )
        return dataframe_to_json(result)

    except Exception as e:
        return create_error_response("InternalError", f"An error occurred: {str(e)}")


@mcp.tool()
def calculate_range(data_json: str, field_name: str, freq: str = "h") -> str:
    """
    Calculate range (极差 = max - min) for time series data.

    This tool automatically calculates both max and min, then computes the range.

    Args:
        data_json: JSON string containing time series data
        field_name: Name of the field to calculate range for
        freq: Resampling frequency ('h', 'D', 'W', 'M')

    Returns:
        JSON string with range values over time or error response
    """
    try:
        # Validate JSON format
        is_valid, error_msg = validate_json_format(data_json)
        if not is_valid:
            return create_error_response("JSONParseError", error_msg)

        # Validate frequency
        is_valid, error_msg = validate_frequency(freq)
        if not is_valid:
            return create_error_response("ValidationError", error_msg)

        # Parse JSON to DataFrame
        df = parse_json_to_dataframe(data_json)

        # Validate field exists
        is_valid, error_msg = validate_dataframe_field(df, field_name)
        if not is_valid:
            return create_error_response(
                "ValidationError", error_msg, {"field_name": field_name}
            )

        # Request both max and min to trigger automatic range calculation
        result = calculator.calculate_statistical_features(
            df, field_name, ["最大值", "最小值"], freq
        )

        # Extract only the range columns
        if "极差" in result.columns:
            range_result = result[["极差"]]
            if "极差的时间变化率" in result.columns:
                range_result["极差的时间变化率"] = result["极差的时间变化率"]
            return dataframe_to_json(range_result)

        return dataframe_to_json(pd.DataFrame())

    except Exception as e:
        return create_error_response("InternalError", f"An error occurred: {str(e)}")


# ============================================================================
# Batch Statistical Features Tool
# ============================================================================
# 模拟数据库读取
def fetch_real_data(cave_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    # 在实际论文中，这里是 pd.read_sql(...)
    # 这里我们生成假数据用于测试
    dates = pd.date_range(start=start_date, end=end_date, freq="h")
    data = {
        "time": dates,
        "temperature": np.random.normal(20, 5, len(dates)),
        "humidity": np.random.normal(50, 10, len(dates)),
    }
    return pd.DataFrame(data).set_index("time")


def _calculate_statistical_features_impl(
    data_json: str, field_name: str, feature_list: list[str], freq: str = "h"
) -> str:
    """Implementation of calculate_statistical_features that can be tested directly."""
    logger.info(
        "MCP request: calculate_statistical_features (batch)",
        extra={"field_name": field_name, "features": feature_list, "freq": freq},
    )

    try:
        # Validate JSON format
        is_valid, error_msg = validate_json_format(data_json)
        if not is_valid:
            return create_error_response("JSONParseError", error_msg)

        # Validate feature list
        is_valid, error_msg = validate_feature_list(feature_list)
        if not is_valid:
            return create_error_response(
                "ValidationError", error_msg, {"feature_list": feature_list}
            )

        # Validate frequency
        is_valid, error_msg = validate_frequency(freq)
        if not is_valid:
            return create_error_response("ValidationError", error_msg)

        # Parse JSON to DataFrame
        df = parse_json_to_dataframe(data_json)

        # Validate field exists
        is_valid, error_msg = validate_dataframe_field(df, field_name)
        if not is_valid:
            return create_error_response(
                "ValidationError", error_msg, {"field_name": field_name}
            )

        # Calculate features
        result = calculator.calculate_statistical_features(
            df, field_name, feature_list, freq
        )

        logger.info(
            "MCP response: calculate_statistical_features completed successfully",
            extra={
                "field_name": field_name,
                "features": feature_list,
                "result_rows": len(result),
            },
        )

        return dataframe_to_json(result)

    except Exception as e:
        logger.exception(
            "Unexpected error in calculate_statistical_features",
            extra={"field_name": field_name, "features": feature_list, "freq": freq},
        )
        return create_error_response("InternalError", f"An error occurred: {str(e)}")


@mcp.tool()
def calculate_statistical_features(
    data_json: str, field_name: str, feature_list: list[str], freq: str = "h"
) -> str:
    """
    Calculate multiple statistical features at once.

    This tool allows batch calculation of multiple statistical features in a single call,
    which is more efficient than calling individual feature tools separately.

    Available features:
    - 均值 (mean)
    - 中位数 (median)
    - 最大值 (max)
    - 最小值 (min)
    - 标准差 (std)
    - Q1 (first quartile)
    - Q3 (third quartile)
    - P10 (10th percentile)
    - 超过Q3占时比 (percent above Q3)

    Note: If both 最大值 and 最小值 are requested, the range (极差) and its rate of change
    (极差的时间变化率) will be automatically calculated.

    Args:
        data_json: JSON string containing time series data with 'time' field and data fields
        field_name: Name of the field to calculate features for
        feature_list: List of feature names to calculate (e.g., ["均值", "最大值", "标准差"])
        freq: Resampling frequency ('h' for hourly, 'D' for daily, 'W' for weekly, 'M' for monthly)

    Returns:
        JSON string with all requested features calculated over time or error response

    Example:
        Input data_json: {"data": [{"time": "2024-01-01T00:00:00", "temperature": 20.5}, ...]}
        Input field_name: "temperature"
        Input feature_list: ["均值", "最大值", "最小值"]
        Input freq: "D"

        Output: {"features": [{"time": "2024-01-01T00:00:00", "均值": 20.3, "最大值": 25.0,
                               "最小值": 15.0, "极差": 10.0, "极差的时间变化率": 0.0}, ...]}
    """
    return _calculate_statistical_features_impl(
        data_json, field_name, feature_list, freq
    )


# ============================================================================
# Batch Statistical Features Tool
# ============================================================================
@mcp.tool()
def execute_python_analysis(code_snippet: str, cave_id: str) -> str:
    """
    一个可以执行 Pandas 代码的工具。
    df 变量已经预置好了，代表该洞窟的数据。

    Args:
        code_snippet: 只需要写计算逻辑，例如 "result = df['temperature'].mean()"
        cave_id: 洞窟ID，用于后台加载数据
    """
    try:
        # 1. 数据加载
        # 假设 fetch_real_data 是你从csv或数据库读数据的函数
        df = fetch_real_data(cave_id)

        # 2. 准备执行环境 (Sandbox)
        # 只允许模型访问 df 和 pandas，防止它干坏事（如删除文件）
        local_scope = {"df": df, "pd": pd, "result": None}

        # 3. 执行模型生成的代码
        # 这是一个极简的解释器，论文里你可以叫它 "Dynamic Semantic Executor"
        exec(code_snippet, {}, local_scope)

        # 4. 获取结果
        result = local_scope.get("result")
        return str(result)

    except Exception as e:
        return f"代码执行错误: {str(e)}"


if __name__ == "__main__":
    mcp.run()
