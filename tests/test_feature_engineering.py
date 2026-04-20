"""Tests for feature engineering module."""

import unittest

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.feature_engineering import (
    TelecomFeatureEngineer,
    FeatureSelector,
    create_preprocessing_pipeline
)


class TestTelecomFeatureEngineer(unittest.TestCase):
    """Test feature engineering functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_df = pd.DataFrame({
            "recharge_amount_m6": [500, 450, 300],
            "recharge_amount_m7": [450, 400, 250],
            "recharge_amount_m8": [300, 200, 150],
            "total_calls_m6": [150, 140, 80],
            "total_calls_m7": [140, 130, 70],
            "total_calls_m8": [80, 50, 30],
            "total_duration_m6": [3000, 2800, 1500],
            "total_duration_m7": [2800, 2600, 1200],
            "total_duration_m8": [1500, 800, 400],
            "incoming_calls_m6": [50, 45, 20],
            "incoming_calls_m7": [45, 40, 15],
            "incoming_calls_m8": [20, 10, 5],
            "outgoing_calls_m6": [100, 95, 60],
            "outgoing_calls_m7": [95, 90, 55],
            "outgoing_calls_m8": [60, 40, 25],
            "data_usage_m6": [2048, 1800, 512],
            "data_usage_m7": [1800, 1500, 256],
            "data_usage_m8": [512, 200, 0],
        })
    
    def test_fit_transform(self):
        """Test fit and transform."""
        engineer = TelecomFeatureEngineer()
        result = engineer.fit_transform(self.sample_df)
        
        # Check output is DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check features were created
        self.assertGreater(len(result.columns), 0)
        
        # Check no NaN values
        self.assertFalse(result.isnull().any().any())
    
    def test_usage_features(self):
        """Test usage feature creation."""
        engineer = TelecomFeatureEngineer()
        engineer.fit(self.sample_df)
        result = engineer._create_usage_features(self.sample_df.copy())
        
        # Check key features exist
        self.assertIn("total_calls_m8", result.columns)
        self.assertIn("total_duration_m8", result.columns)
    
    def test_trend_features(self):
        """Test trend feature creation."""
        engineer = TelecomFeatureEngineer()
        result = engineer._create_trend_features(self.sample_df.copy())
        
        # Check trend features exist
        trend_cols = [col for col in result.columns if "trend" in col]
        self.assertGreater(len(trend_cols), 0)


class TestFeatureSelector(unittest.TestCase):
    """Test feature selector."""
    
    def setUp(self):
        """Set up test data."""
        self.X = pd.DataFrame({
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "feature_3": np.ones(100) * 0.001,  # Low variance
            "feature_4": np.random.randn(100),
        })
        self.X["feature_4"] = self.X["feature_1"] * 0.99  # Highly correlated
    
    def test_feature_selection(self):
        """Test feature selection removes low variance/correlated features."""
        selector = FeatureSelector(
            correlation_threshold=0.95,
            variance_threshold=0.01
        )
        result = selector.fit_transform(self.X)
        
        # Should reduce number of features
        self.assertLess(len(result.columns), len(self.X.columns))


class TestPreprocessingPipeline(unittest.TestCase):
    """Test complete preprocessing pipeline."""
    
    def test_pipeline_creation(self):
        """Test pipeline can be created."""
        pipeline = create_preprocessing_pipeline()
        self.assertIsNotNone(pipeline)


if __name__ == "__main__":
    unittest.main()
