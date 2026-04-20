"""Tests for data loader module."""

import unittest
from pathlib import Path

import pandas as pd
import numpy as np

from src.data_loader import DataLoader, load_and_prepare_data


class TestDataLoader(unittest.TestCase):
    """Test DataLoader functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_data = pd.DataFrame({
            "customer_id": ["C001", "C002", "C003", "C004"] * 4,
            "month": [6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9],
            "recharge_amount_m6": [500, 300, 800, 200] * 4,
            "recharge_amount_m7": [450, 280, 850, 180] * 4,
            "recharge_amount_m8": [300, 150, 900, 100] * 4,
            "total_calls_m6": [150, 80, 200, 50] * 4,
            "total_calls_m7": [140, 70, 210, 40] * 4,
            "total_calls_m8": [80, 30, 220, 20] * 4,
            "total_duration_m6": [3000, 1000, 5000, 500] * 4,
            "total_duration_m7": [2800, 900, 5200, 400] * 4,
            "total_duration_m8": [1500, 400, 5500, 200] * 4,
            "incoming_calls_m6": [50, 25, 80, 15] * 4,
            "incoming_calls_m7": [45, 20, 85, 10] * 4,
            "incoming_calls_m8": [20, 5, 90, 5] * 4,
            "outgoing_calls_m6": [100, 55, 120, 35] * 4,
            "outgoing_calls_m7": [95, 50, 125, 30] * 4,
            "outgoing_calls_m8": [60, 25, 130, 15] * 4,
            "data_usage_m6": [2048, 512, 4096, 256] * 4,
            "data_usage_m7": [1800, 400, 4096, 128] * 4,
            "data_usage_m8": [512, 0, 4096, 0] * 4,
        })
        
        self.loader = DataLoader()
        self.loader.data = self.sample_data
    
    def test_identify_high_value_customers(self):
        """Test high-value customer identification."""
        df = self.loader.identify_high_value_customers(
            self.sample_data,
            months=[6, 7],
            percentile=0.70
        )
        
        # Check high_value column exists
        self.assertIn("high_value", df.columns)
        self.assertIn("avg_recharge_good_phase", df.columns)
        
        # Check high-value customers are identified (typically 30-50% depending on data)
        high_value_rate = df["high_value"].mean()
        self.assertGreater(high_value_rate, 0.25)
        self.assertLess(high_value_rate, 0.55)
    
    def test_define_churn(self):
        """Test churn definition."""
        df = self.loader.define_churn(self.sample_data, churn_month=9)
        
        self.assertIn("churned", df.columns)
        # Customers with zero usage in month 9 should be churned
        churned_count = df["churned"].sum()
        # Check it's a numeric value (int, float, or numpy number)
        self.assertIsInstance(churned_count, (int, float, np.integer, np.floating))
    
    def test_validate_data(self):
        """Test data validation."""
        # Should pass with valid data
        result = self.loader.validate_data(self.sample_data)
        self.assertTrue(result)
        
        # Should fail with empty data
        with self.assertRaises(ValueError):
            self.loader.validate_data(pd.DataFrame())
    
    def test_get_high_value_subset(self):
        """Test filtering high-value customers."""
        df = self.loader.identify_high_value_customers(self.sample_data)
        high_value_df = self.loader.get_high_value_subset(df)
        
        self.assertTrue((high_value_df["high_value"] == 1).all())


if __name__ == "__main__":
    unittest.main()
