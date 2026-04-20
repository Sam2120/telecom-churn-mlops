#!/usr/bin/env python
"""Split data into train and test sets."""

import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_DIR, RANDOM_STATE, TEST_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Split data into train and test sets."""
    logger.info("Splitting data...")
    
    # Load processed data
    X = pd.read_csv(DATA_DIR / "processed" / "features.csv")
    y = pd.read_csv(DATA_DIR / "processed" / "target.csv").squeeze()
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Train churn rate: {y_train.mean():.2%}")
    logger.info(f"Test churn rate: {y_test.mean():.2%}")
    
    # Save splits
    X_train.to_csv(DATA_DIR / "processed" / "X_train.csv", index=False)
    X_test.to_csv(DATA_DIR / "processed" / "X_test.csv", index=False)
    y_train.to_csv(DATA_DIR / "processed" / "y_train.csv", index=False)
    y_test.to_csv(DATA_DIR / "processed" / "y_test.csv", index=False)
    
    logger.info("Data split complete")


if __name__ == "__main__":
    main()
