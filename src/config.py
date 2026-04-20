"""Configuration settings for the churn prediction pipeline."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Data Paths
RAW_DATA_PATH = DATA_DIR / "raw" / "telecom_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "processed_data.csv"
FEATURES_PATH = DATA_DIR / "processed" / "features.csv"

# Model Paths
MODEL_PATH = MODELS_DIR / "churn_model.pkl"
PCA_MODEL_PATH = MODELS_DIR / "pca_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
INTERPRETABLE_MODEL_PATH = MODELS_DIR / "interpretable_model.pkl"

# AWS S3 Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "telecom-churn-bucket")

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "telecom_churn_prediction")

# Hugging Face Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "telecom-churn-model")

# Model Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
HIGH_VALUE_THRESHOLD = 0.70  # 70th percentile for high-value customers

# Feature Engineering Configuration
GOOD_PHASE_MONTHS = [6, 7]  # Months for identifying high-value customers
ACTION_PHASE_MONTH = 8  # Month for prediction
CHURN_PHASE_MONTH = 9  # Month for churn definition

# Class Imbalance Configuration
USE_SMOTE = True
SMOTE_SAMPLING_STRATEGY = 0.5
CLASS_WEIGHT = "balanced"

# PCA Configuration
APPLY_PCA = True
PCA_VARIANCE_THRESHOLD = 0.95
PCA_N_COMPONENTS = None  # Will be determined by variance threshold

# Model Hyperparameters
XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

LOGISTIC_REGRESSION_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 1000,
    "class_weight": CLASS_WEIGHT,
    "random_state": RANDOM_STATE,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": CLASS_WEIGHT,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_DEBUG = os.getenv("API_DEBUG", "false").lower() == "true"

# Feature Groups
USAGE_FEATURES = [
    "total_calls_m6", "total_calls_m7", "total_calls_m8",
    "total_duration_m6", "total_duration_m7", "total_duration_m8",
    "incoming_calls_m6", "incoming_calls_m7", "incoming_calls_m8",
    "outgoing_calls_m6", "outgoing_calls_m7", "outgoing_calls_m8",
]

RECHARGE_FEATURES = [
    "recharge_amount_m6", "recharge_amount_m7", "recharge_amount_m8",
    "recharge_count_m6", "recharge_count_m7", "recharge_count_m8",
    "avg_recharge_amount",
]

INTERNET_FEATURES = [
    "data_usage_m6", "data_usage_m7", "data_usage_m8",
    "data_sessions_m6", "data_sessions_m7", "data_sessions_m8",
]

# Ensure directories exist
for directory in [DATA_DIR / "raw", DATA_DIR / "processed", DATA_DIR / "interim", 
                  MODELS_DIR, ARTIFACTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
