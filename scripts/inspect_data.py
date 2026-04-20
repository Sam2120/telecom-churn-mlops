#!/usr/bin/env python
"""Inspect and adapt existing telecom data to required format."""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RAW_DATA_PATH


def inspect_data():
    """Inspect the raw data file and suggest column mappings."""
    print(f"Inspecting: {RAW_DATA_PATH}")
    
    # Load a sample
    df = pd.read_csv(RAW_DATA_PATH, nrows=5)
    
    print(f"\nShape: {df.shape}")
    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    print(f"\nSample data:")
    print(df.head())
    
    # Check for potential ID columns
    print("\n" + "="*50)
    print("COLUMN MAPPING SUGGESTIONS:")
    print("="*50)
    
    # Look for ID-like columns
    id_candidates = [col for col in df.columns if any(keyword in col.lower() 
                     for keyword in ['id', 'customer', 'user', 'subscriber', 'msisdn', 'phone'])]
    print(f"\nPotential ID columns: {id_candidates}")
    
    # Look for month/date columns
    month_candidates = [col for col in df.columns if any(keyword in col.lower() 
                       for keyword in ['month', 'date', 'period', 'time', 'm6', 'm7', 'm8', 'm9'])]
    print(f"Potential month/date columns: {month_candidates}")
    
    # Look for recharge columns
    recharge_candidates = [col for col in df.columns if any(keyword in col.lower() 
                          for keyword in ['recharge', 'amount', 'balance', 'topup'])]
    print(f"Potential recharge columns: {recharge_candidates}")
    
    # Look for call columns
    call_candidates = [col for col in df.columns if any(keyword in col.lower() 
                      for keyword in ['call', 'voice', 'minutes', 'duration'])]
    print(f"Potential call columns: {call_candidates}")
    
    # Look for data columns
    data_candidates = [col for col in df.columns if any(keyword in col.lower() 
                     for keyword in ['data', 'internet', 'mb', 'gb', 'usage'])]
    print(f"Potential data columns: {data_candidates}")
    
    return df.columns.tolist()


def adapt_data(column_mapping=None):
    """Adapt existing data to required format.
    
    Args:
        column_mapping: Dict mapping existing columns to required columns
                       e.g., {'subscriber_id': 'customer_id', 'period': 'month'}
    """
    print(f"\nAdapting data...")
    
    # Load full dataset
    df = pd.read_csv(RAW_DATA_PATH)
    
    if column_mapping:
        # Rename columns
        df = df.rename(columns=column_mapping)
        print(f"Renamed columns: {column_mapping}")
    
    # Ensure required columns exist
    required = ['customer_id', 'month']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        print(f"\nERROR: Still missing required columns: {missing}")
        print("Please provide a column_mapping to fix this.")
        return None
    
    # Save adapted data
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"\nSaved adapted data to: {RAW_DATA_PATH}")
    print(f"Final columns: {df.columns.tolist()[:10]}...")
    
    return df


if __name__ == "__main__":
    columns = inspect_data()
    
    print("\n" + "="*50)
    print("NEXT STEPS:")
    print("="*50)
    print("""
1. Identify your column names from the list above

2. Create a column mapping in this script or run:
   
   python -c "
   from scripts.inspect_data import adapt_data
   mapping = {
       'your_id_column': 'customer_id',
       'your_month_column': 'month',
       # Add other mappings as needed
   }
   adapt_data(mapping)
   "

3. Or manually rename columns in your CSV before processing
    """)
