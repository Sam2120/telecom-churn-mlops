#!/usr/bin/env python
"""Load and validate raw data."""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_and_prepare_data
from src.config import DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Load and prepare data for the pipeline."""
    logger.info("Loading raw data...")
    
    # Load data with high-value identification and churn definition
    df = load_and_prepare_data(
        identify_high_value=True,
        define_churn=True,
        high_value_only=True
    )
    
    # Save interim data
    output_path = DATA_DIR / "interim" / "raw_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Data saved to {output_path}")
    logger.info(f"Total records: {len(df)}")
    logger.info(f"Churn rate: {df['churned'].mean():.2%}")
    
    return df


if __name__ == "__main__":
    main()
