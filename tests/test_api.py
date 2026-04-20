"""Tests for FastAPI application."""

import unittest
from fastapi.testclient import TestClient

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app


class TestAPI(unittest.TestCase):
    """Test API endpoints."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test client."""
        cls.client = TestClient(app)
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())
    
    def test_health_check(self):
        """Test health endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("model_loaded", data)
    
    def test_predict_endpoint_structure(self):
        """Test predict endpoint structure validation."""
        # Test with missing required fields
        response = self.client.post("/predict", json={})
        self.assertEqual(response.status_code, 422)  # Validation error
    
    def test_predict_with_valid_data(self):
        """Test predict with valid customer data."""
        customer_data = {
            "customer_id": "TEST001",
            "recharge_amount_m6": 500,
            "recharge_amount_m7": 450,
            "recharge_amount_m8": 300,
            "total_calls_m6": 150,
            "total_calls_m7": 140,
            "total_calls_m8": 80,
            "total_duration_m6": 3000,
            "total_duration_m7": 2800,
            "total_duration_m8": 1500,
            "incoming_calls_m6": 50,
            "incoming_calls_m7": 45,
            "incoming_calls_m8": 20,
            "outgoing_calls_m6": 100,
            "outgoing_calls_m7": 95,
            "outgoing_calls_m8": 60,
            "data_usage_m6": 2048,
            "data_usage_m7": 1800,
            "data_usage_m8": 512
        }
        
        response = self.client.post("/predict", json=customer_data)
        # May return 200 or 503 depending on model loading
        self.assertIn(response.status_code, [200, 503])
    
    def test_model_info_endpoint(self):
        """Test model info endpoint."""
        response = self.client.get("/model/info")
        # May return 200 or 503 depending on model loading
        self.assertIn(response.status_code, [200, 503])
    
    def test_batch_predict_structure(self):
        """Test batch predict structure."""
        response = self.client.post("/predict/batch", json={"customers": []})
        self.assertIn(response.status_code, [200, 503])


if __name__ == "__main__":
    unittest.main()
