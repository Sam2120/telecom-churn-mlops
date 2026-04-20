#!/usr/bin/env python
"""Preprocess and engineer features."""

import logging
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering import TelecomFeatureEngineer, FeatureSelector
from src.config import DATA_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Preprocess data and engineer features."""
    logger.info("Starting preprocessing...")
    
    # Load raw data
    input_path = DATA_DIR / "interim" / "raw_data.csv"
    df = pd.read_csv(input_path)
    
    # Separate target and IDs
    target = df["churned"].copy()
    customer_ids = df["customer_id"].copy() if "customer_id" in df.columns else None
    
    # Drop non-feature columns
    cols_to_drop = ["churned", "high_value", "month", "avg_recharge_good_phase"]
    if "customer_id" in df.columns:
        cols_to_drop.append("customer_id")
    
    features_df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    
    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    numeric_cols = features_df.select_dtypes(include=["float64", "int64"]).columns
    features_df[numeric_cols] = imputer.fit_transform(features_df[numeric_cols])
    
    # Engineer features
    engineer = TelecomFeatureEngineer(action_month=8)
    engineered_features = engineer.transform(features_df)
    
    # Select features
    selector = FeatureSelector()
    X = selector.fit_transform(engineered_features)
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Selected features: {len(selector.selected_features)}")
    
    # Save processed data
    X.to_csv(DATA_DIR / "processed" / "features.csv", index=False)
    target.to_csv(DATA_DIR / "processed" / "target.csv", index=False)
    
    if customer_ids is not None:
        customer_ids.to_csv(DATA_DIR / "processed" / "customer_ids.csv", index=False)
    
    # Save preprocessor
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    preprocessor = {
        "imputer": imputer,
        "engineer": engineer,
        "selector": selector,
        "feature_names": X.columns.tolist()
    }
    with open(MODELS_DIR / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    
    logger.info("Preprocessing complete")


if __name__ == "__main__":
    main()
