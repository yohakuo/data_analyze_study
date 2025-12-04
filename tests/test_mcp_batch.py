"""
Test for MCP batch statistical features tool.

This test verifies that the batch statistical features MCP tool works correctly.
"""

import json
import pandas as pd
import pytest
from datetime import datetime, timezone

# Import the calculator and helper functions directly
from src.features.calculator import FeatureCalculator
from src.features.mcp_server import parse_json_to_dataframe, dataframe_to_json


def test_batch_statistical_features_basic():
    """
    Test that the batch statistical features tool accepts parameters and returns results.
    
    This test verifies Requirements 4.2 and 4.7:
    - 4.2: MCP accepts JSON format and returns JSON format
    - 4.7: Batch calculation tools compute multiple features at once
    """
    # Create sample time series data
    dates = pd.date_range(start="2024-01-01", periods=24, freq="h", tz=timezone.utc)
    data = {
        "time": [d.isoformat() for d in dates],
        "temperature": [20 + i * 0.5 for i in range(24)]
    }
    
    # Convert to JSON string
    data_json = json.dumps({"data": data})
    
    # Simulate what the MCP tool does: parse JSON, call calculator, return JSON
    calculator = FeatureCalculator()
    df = parse_json_to_dataframe(data_json)
    result_df = calculator.calculate_statistical_features(
        df=df,
        field_name="temperature",
        feature_list=["均值", "最大值", "最小值", "标准差"],
        freq="D"
    )
    result_json = dataframe_to_json(result_df)
    
    # Parse result
    result = json.loads(result_json)
    
    # Verify result structure
    assert "features" in result, "Result should contain 'features' key"
    assert len(result["features"]) > 0, "Result should contain at least one feature row"
    
    # Verify requested features are present
    first_row = result["features"][0]
    assert "均值" in first_row, "Result should contain 均值"
    assert "最大值" in first_row, "Result should contain 最大值"
    assert "最小值" in first_row, "Result should contain 最小值"
    assert "标准差" in first_row, "Result should contain 标准差"
    
    print("✓ Batch statistical features tool works correctly")


def test_batch_with_range_calculation():
    """
    Test that requesting both max and min triggers automatic range calculation.
    
    This verifies that the batch tool properly handles the automatic range
    calculation feature when both 最大值 and 最小值 are requested.
    """
    # Create sample time series data with varying values
    dates = pd.date_range(start="2024-01-01", periods=48, freq="h", tz=timezone.utc)
    data = {
        "time": [d.isoformat() for d in dates],
        "temperature": [20 + (i % 12) for i in range(48)]  # Creates a pattern
    }
    
    # Convert to JSON string
    data_json = json.dumps({"data": data})
    
    # Simulate what the MCP tool does
    calculator = FeatureCalculator()
    df = parse_json_to_dataframe(data_json)
    result_df = calculator.calculate_statistical_features(
        df=df,
        field_name="temperature",
        feature_list=["最大值", "最小值"],
        freq="D"
    )
    result_json = dataframe_to_json(result_df)
    
    # Parse result
    result = json.loads(result_json)
    
    # Verify range columns are present
    first_row = result["features"][0]
    assert "极差" in first_row, "Result should contain 极差 (range)"
    assert "极差的时间变化率" in first_row, "Result should contain 极差的时间变化率"
    
    # Verify range calculation is correct
    assert first_row["极差"] == first_row["最大值"] - first_row["最小值"], \
        "Range should equal max - min"
    
    print("✓ Automatic range calculation works in batch tool")


if __name__ == "__main__":
    test_batch_statistical_features_basic()
    test_batch_with_range_calculation()
    print("\n✓ All batch statistical features tests passed!")



# ============================================================================
# Unit Tests for MCP Error Response Format
# ============================================================================

def test_error_response_structure():
    """
    Test that error responses have the correct structure.
    
    This test verifies Requirements 4.4: structured error response format
    with type, message, and details fields.
    """
    from src.features.mcp_server import create_error_response
    
    # Create an error response
    error_response = create_error_response(
        error_type="ValidationError",
        message="Field 'temperature' not found in DataFrame",
        details={"field_name": "temperature", "available_fields": ["humidity", "pressure"]}
    )
    
    # Parse the JSON response
    error_data = json.loads(error_response)
    
    # Verify top-level structure
    assert "success" in error_data, "Error response should have 'success' field"
    assert error_data["success"] is False, "Error response should have success=False"
    assert "error" in error_data, "Error response should have 'error' field"
    
    # Verify error object structure
    error_obj = error_data["error"]
    assert "type" in error_obj, "Error object should have 'type' field"
    assert "message" in error_obj, "Error object should have 'message' field"
    assert "details" in error_obj, "Error object should have 'details' field"
    
    # Verify error content
    assert error_obj["type"] == "ValidationError", "Error type should match"
    assert error_obj["message"] == "Field 'temperature' not found in DataFrame", "Error message should match"
    assert error_obj["details"]["field_name"] == "temperature", "Error details should match"
    
    print("✓ Error response structure is correct")


def test_invalid_json_returns_error():
    """
    Test that invalid JSON input returns a structured error response.
    
    This verifies that JSON parsing errors are caught and returned
    as structured error responses.
    """
    from src.features.mcp_server import _calculate_mean_impl
    
    # Test with invalid JSON
    invalid_json = "{this is not valid json"
    result = _calculate_mean_impl(invalid_json, "temperature", "h")
    
    # Parse result
    result_data = json.loads(result)
    
    # Verify it's an error response
    assert "success" in result_data, "Response should have 'success' field"
    assert result_data["success"] is False, "Invalid JSON should result in error"
    assert "error" in result_data, "Response should have 'error' field"
    assert result_data["error"]["type"] == "JSONParseError", "Error type should be JSONParseError"
    assert "message" in result_data["error"], "Error should have message"
    assert "details" in result_data["error"], "Error should have details"
    
    print("✓ Invalid JSON returns structured error response")


def test_missing_field_returns_error():
    """
    Test that requesting a non-existent field returns a structured error response.
    
    This verifies that field validation errors are caught and returned
    as structured error responses with helpful details.
    """
    from src.features.mcp_server import _calculate_mean_impl
    
    # Create valid JSON with a field
    dates = pd.date_range(start="2024-01-01", periods=24, freq="h", tz=timezone.utc)
    data = {
        "time": [d.isoformat() for d in dates],
        "humidity": [60 + i * 0.5 for i in range(24)]
    }
    data_json = json.dumps({"data": data})
    
    # Request a field that doesn't exist
    result = _calculate_mean_impl(data_json, "temperature", "h")
    
    # Parse result
    result_data = json.loads(result)
    
    # Verify it's an error response
    assert "success" in result_data, "Response should have 'success' field"
    assert result_data["success"] is False, "Missing field should result in error"
    assert "error" in result_data, "Response should have 'error' field"
    assert result_data["error"]["type"] == "ValidationError", "Error type should be ValidationError"
    assert "temperature" in result_data["error"]["message"], "Error message should mention the missing field"
    assert "not found" in result_data["error"]["message"].lower(), "Error message should indicate field not found"
    assert "details" in result_data["error"], "Error should have details"
    
    print("✓ Missing field returns structured error response")


def test_invalid_frequency_returns_error():
    """
    Test that an invalid frequency parameter returns a structured error response.
    
    This verifies that parameter validation errors are caught and returned
    as structured error responses.
    """
    from src.features.mcp_server import _calculate_mean_impl
    
    # Create valid JSON
    dates = pd.date_range(start="2024-01-01", periods=24, freq="h", tz=timezone.utc)
    data = {
        "time": [d.isoformat() for d in dates],
        "temperature": [20 + i * 0.5 for i in range(24)]
    }
    data_json = json.dumps({"data": data})
    
    # Use invalid frequency
    result = _calculate_mean_impl(data_json, "temperature", "INVALID")
    
    # Parse result
    result_data = json.loads(result)
    
    # Verify it's an error response
    assert "success" in result_data, "Response should have 'success' field"
    assert result_data["success"] is False, "Invalid frequency should result in error"
    assert "error" in result_data, "Response should have 'error' field"
    assert result_data["error"]["type"] == "ValidationError", "Error type should be ValidationError"
    assert "frequency" in result_data["error"]["message"].lower(), "Error message should mention frequency"
    
    print("✓ Invalid frequency returns structured error response")


def test_invalid_feature_list_returns_error():
    """
    Test that an invalid feature list returns a structured error response.
    
    This verifies that feature list validation errors are caught and returned
    as structured error responses.
    """
    from src.features.mcp_server import _calculate_statistical_features_impl
    
    # Create valid JSON
    dates = pd.date_range(start="2024-01-01", periods=24, freq="h", tz=timezone.utc)
    data = {
        "time": [d.isoformat() for d in dates],
        "temperature": [20 + i * 0.5 for i in range(24)]
    }
    data_json = json.dumps({"data": data})
    
    # Use invalid feature list
    result = _calculate_statistical_features_impl(data_json, "temperature", ["invalid_feature", "another_invalid"], "h")
    
    # Parse result
    result_data = json.loads(result)
    
    # Verify it's an error response
    assert "success" in result_data, "Response should have 'success' field"
    assert result_data["success"] is False, "Invalid feature list should result in error"
    assert "error" in result_data, "Response should have 'error' field"
    assert result_data["error"]["type"] == "ValidationError", "Error type should be ValidationError"
    assert "invalid" in result_data["error"]["message"].lower(), "Error message should mention invalid features"
    
    print("✓ Invalid feature list returns structured error response")


def test_empty_json_returns_error():
    """
    Test that empty JSON input returns a structured error response.
    
    This verifies that empty input validation is performed.
    """
    from src.features.mcp_server import _calculate_mean_impl
    
    # Test with empty string
    result = _calculate_mean_impl("", "temperature", "h")
    
    # Parse result
    result_data = json.loads(result)
    
    # Verify it's an error response
    assert "success" in result_data, "Response should have 'success' field"
    assert result_data["success"] is False, "Empty JSON should result in error"
    assert "error" in result_data, "Response should have 'error' field"
    assert result_data["error"]["type"] == "JSONParseError", "Error type should be JSONParseError"
    assert "empty" in result_data["error"]["message"].lower(), "Error message should mention empty input"
    
    print("✓ Empty JSON returns structured error response")


def test_error_response_with_no_details():
    """
    Test that error responses work correctly when no details are provided.
    
    This verifies that the details field defaults to an empty dict.
    """
    from src.features.mcp_server import create_error_response
    
    # Create error response without details
    error_response = create_error_response(
        error_type="InternalError",
        message="An unexpected error occurred"
    )
    
    # Parse the JSON response
    error_data = json.loads(error_response)
    
    # Verify structure
    assert error_data["success"] is False
    assert error_data["error"]["type"] == "InternalError"
    assert error_data["error"]["message"] == "An unexpected error occurred"
    assert "details" in error_data["error"], "Details field should exist"
    assert error_data["error"]["details"] == {}, "Details should be empty dict when not provided"
    
    print("✓ Error response works with no details")
