#!/usr/bin/env python
"""Generate sample telecom data for testing the pipeline."""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_DIR, RAW_DATA_PATH


def generate_sample_data(n_customers=1000, random_state=42):
    """Generate synthetic telecom customer data.
    
    Args:
        n_customers: Number of customers to generate
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with telecom data
    """
    np.random.seed(random_state)
    
    data = []
    
    for month in [6, 7, 8, 9]:  # 4 months of data
        for i in range(n_customers):
            customer_id = f"CUST{i+1:05d}"
            
            # Base values with some randomness
            base_recharge = np.random.exponential(300, 1)[0] + 100
            base_calls = np.random.poisson(100)
            base_duration = base_calls * np.random.exponential(20, 1)[0]
            base_data = np.random.exponential(1024, 1)[0]
            
            # Simulate declining usage leading to churn in month 9
            if month == 9 and i < n_customers * 0.2:  # 20% churners
                recharge = 0
                calls = 0
                duration = 0
                data_usage = 0
                incoming = 0
                outgoing = 0
            elif month == 8 and i < n_customers * 0.2:  # Pre-churn decline
                factor = np.random.uniform(0.1, 0.5)
                recharge = base_recharge * factor
                calls = int(base_calls * factor)
                duration = base_duration * factor
                data_usage = base_data * factor
                incoming = int(calls * np.random.uniform(0.2, 0.4))
                outgoing = calls - incoming
            else:
                # Normal usage with month-to-month variation
                variation = np.random.uniform(0.8, 1.2)
                recharge = base_recharge * variation
                calls = int(base_calls * variation)
                duration = base_duration * variation
                data_usage = base_data * variation
                incoming = int(calls * np.random.uniform(0.3, 0.5))
                outgoing = calls - incoming
            
            record = {
                "customer_id": customer_id,
                "month": month,
                f"recharge_amount_m{month}": max(0, round(recharge, 2)),
                f"total_calls_m{month}": max(0, calls),
                f"total_duration_m{month}": max(0, round(duration, 2)),
                f"incoming_calls_m{month}": max(0, incoming),
                f"outgoing_calls_m{month}": max(0, outgoing),
                f"data_usage_m{month}": max(0, round(data_usage, 2)),
            }
            data.append(record)
    
    # Convert to DataFrame and pivot
    df = pd.DataFrame(data)
    
    # Pivot to have one row per customer with columns for each month
    pivot_data = {}
    for month in [6, 7, 8, 9]:
        month_data = df[df["month"] == month].copy()
        month_data = month_data.drop("month", axis=1)
        month_data = month_data.set_index("customer_id")
        
        # Rename columns to include month
        month_data = month_data.rename(columns={
            col: f"{col}" for col in month_data.columns
        })
        
        pivot_data[month] = month_data
    
    # Combine all months
    combined = pivot_data[6].join(pivot_data[7], lsuffix='', rsuffix='_m7', how='outer')
    combined = combined.join(pivot_data[8], rsuffix='_m8', how='outer')
    combined = combined.join(pivot_data[9], rsuffix='_m9', how='outer')
    
    # Reset index to get customer_id as column
    combined = combined.reset_index()
    combined = combined.rename(columns={"customer_id": "customer_id"})
    
    # Flatten - create one row per customer-month
    records = []
    for _, row in combined.iterrows():
        customer_id = row["customer_id"]
        for month in [6, 7, 8, 9]:
            month_cols = [col for col in combined.columns if f"_m{month}" in col or col == "customer_id"]
            if month_cols:
                record = {"customer_id": customer_id, "month": month}
                
                # Add month-specific columns
                for col in combined.columns:
                    if f"_m{month}" in col:
                        base_col = col.replace(f"_m{month}", "")
                        record[base_col] = row[col] if not pd.isna(row[col]) else 0
                    elif col not in ["customer_id"] and "_m" not in col and month == 6:
                        # Month 6 data without suffix
                        record[col] = row[col] if not pd.isna(row[col]) else 0
                
                records.append(record)
    
    # Create final DataFrame with proper column names
    final_data = []
    for customer_id in combined["customer_id"].unique():
        for month in [6, 7, 8, 9]:
            row_data = {"customer_id": customer_id, "month": month}
            
            # Get values from combined dataframe
            customer_row = combined[combined["customer_id"] == customer_id].iloc[0]
            
            for metric in ["recharge_amount", "total_calls", "total_duration", 
                          "incoming_calls", "outgoing_calls", "data_usage"]:
                col_name = f"{metric}_m{month}"
                if col_name in customer_row:
                    row_data[metric] = customer_row[col_name] if not pd.isna(customer_row[col_name]) else 0
                else:
                    row_data[metric] = 0
            
            final_data.append(row_data)
    
    result_df = pd.DataFrame(final_data)
    
    # Ensure all required columns exist
    required_cols = ["customer_id", "month", "recharge_amount", "total_calls",
                     "total_duration", "incoming_calls", "outgoing_calls", "data_usage"]
    
    for col in required_cols:
        if col not in result_df.columns:
            result_df[col] = 0
    
    return result_df[required_cols]


def main():
    """Generate and save sample data."""
    print("Generating sample telecom data...")
    
    # Create directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
    
    # Generate data
    df = generate_sample_data(n_customers=1000)
    
    # Save to CSV
    output_path = RAW_DATA_PATH
    df.to_csv(output_path, index=False)
    
    print(f"Generated {len(df)} records ({df['customer_id'].nunique()} customers x 4 months)")
    print(f"Saved to: {output_path}")
    print(f"\nSample data:")
    print(df.head(12))  # Show 3 customers (4 months each)
    
    # Statistics
    print(f"\nStatistics:")
    print(f"- Total customers: {df['customer_id'].nunique()}")
    print(f"- Months covered: {sorted(df['month'].unique())}")
    print(f"- Average recharge amount: ${df['recharge_amount'].mean():.2f}")
    print(f"- Average calls per month: {df['total_calls'].mean():.1f}")


if __name__ == "__main__":
    main()
