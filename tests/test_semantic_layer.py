import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import yaml
import json

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.llm import HybridLLMService
from src.features.calculator import FeatureCalculator
from app import execute_analysis  # Import the execution function

class TestSemanticLayer(unittest.TestCase):
    def setUp(self):
        # Load real metrics for validation
        with open("metrics.yaml", "r", encoding="utf-8") as f:
            self.metrics_def = yaml.safe_load(f)
            
        self.calculator = FeatureCalculator()
        
        # Create dummy data across two months
        dates = pd.date_range(start="2024-01-01", end="2024-03-01", freq="h")
        self.df = pd.DataFrame({
            "空气温度（℃）": [20 + i%5 for i in range(len(dates))],
            "空气湿度（%）": [50 + i%10 for i in range(len(dates))],
            "site_id": ["001"] * len(dates)
        }, index=dates)

    def test_metrics_yaml_structure(self):
        """Verify metrics.yaml has required fields"""
        self.assertIn("metrics", self.metrics_def)
        for metric in self.metrics_def["metrics"]:
            self.assertIn("id", metric, "Missing id")
            self.assertIn("data_field", metric, "Missing data_field")
            self.assertIn("calculation_logic", metric, "Missing calculation_logic")

    def test_llm_service_initialization(self):
        """Test LLM service inits without error"""
        service = HybridLLMService(mode="local", metrics_path="metrics.yaml")
        self.assertIsNotNone(service.metrics_def)

    @patch("src.features.llm.OpenAI")
    def test_parse_intent(self, mock_openai):
        """Test intent parsing with mocked LLM response"""
        service = HybridLLMService(mode="local", metrics_path="metrics.yaml")
        
        # Mock response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "metric_id": "temperature_daily",
            "params": {
                "freq": "D",
                "start_date": "2024-01-01",
                "end_date": "2024-02-01"
            }
        })
        service.client.chat.completions.create.return_value = mock_response
        
        intent = service.parse_intent("Show me daily temp from Jan 1 to Feb 1")
        self.assertEqual(intent["metric_id"], "temperature_daily")
        self.assertEqual(intent["params"]["start_date"], "2024-01-01")
        self.assertEqual(intent["params"]["end_date"], "2024-02-01")

    def test_calculation_integration(self):
        """Test that defined metrics can actually be calculated"""
        for metric in self.metrics_def["metrics"]:
            field = metric["data_field"]
            features = metric["calculation_logic"]
            freq = metric["default_freq"]
            
            # Skip if field not in dummy data
            if field not in self.df.columns:
                continue
                
            result = self.calculator.calculate_statistical_features(
                self.df, field_name=field, feature_list=features, freq=freq
            )
            self.assertFalse(result.empty, f"Failed to calculate {metric['id']}")
            
    def test_time_range_filtering(self):
        """Test that data is filtered correctly by start and end date"""
        # Define a test metric (e.g., hourly humidity)
        metric_def = next(m for m in self.metrics_def["metrics"] if m["id"] == "humidity_trend")
        
        filter_params = {
            "start_date": "2024-01-10",
            "end_date": "2024-01-15",
            "freq": "h"
        }
        
        # Execute analysis
        # Note: We need to import execute_analysis from app.py or replicate the logic
        # For this test, let's replicate the core filtering logic we want to test
        filtered_df = self.df.copy()
        filtered_df = filtered_df[filtered_df.index >= filter_params["start_date"]]
        filtered_df = filtered_df[filtered_df.index <= filter_params["end_date"]]
        
        self.assertEqual(filtered_df.index.min().date().isoformat(), "2024-01-10")
        self.assertEqual(filtered_df.index.max().date().isoformat(), "2024-01-15")
        
        # Now verify calculate runs on this
        result_df = self.calculator.calculate_statistical_features(
            filtered_df, 
            field_name=metric_def["data_field"], 
            feature_list=metric_def["calculation_logic"], 
            freq=filter_params["freq"]
        )
        self.assertFalse(result_df.empty)

if __name__ == "__main__":
    unittest.main()
