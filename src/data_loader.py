"""Data loading and validation utilities."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    GOOD_PHASE_MONTHS,
    HIGH_VALUE_THRESHOLD,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading, validation, and initial processing of telecom data."""
    
    def __init__(self, data_path: Optional[Path] = None):
        """Initialize the DataLoader.
        
        Args:
            data_path: Path to the raw data file. Uses config default if not provided.
        """
        self.data_path = data_path or RAW_DATA_PATH
        self.data: Optional[pd.DataFrame] = None
        
    def load_data(self) -> pd.DataFrame:
        """Load raw data from CSV file.
        
        Returns:
            Raw telecom data DataFrame
            
        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        logger.info(f"Loading data from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        self.data = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.data)} rows and {len(self.data.columns)} columns")
        
        return self.data
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate the loaded data structure.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If validation fails
        """
        logger.info("Validating data structure")
        
        # Check for empty DataFrame
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # Check for required columns - support both long and wide formats
        # Wide format: mobile_number + month-specific columns (_6, _7, _8, _9)
        # Long format: customer_id + month
        has_mobile = "mobile_number" in df.columns
        has_customer_id = "customer_id" in df.columns
        has_month = "month" in df.columns
        has_month_suffix = any(col.endswith("_6") for col in df.columns)
        
        if not (has_mobile or has_customer_id):
            raise ValueError("Missing customer identifier column (need 'mobile_number' or 'customer_id')")
        
        if not (has_month or has_month_suffix):
            raise ValueError("Missing month information (need 'month' column or columns with _6, _7, _8, _9 suffixes)")
        
        # Rename mobile_number to customer_id if needed
        if has_mobile and not has_customer_id:
            df.rename(columns={"mobile_number": "customer_id"}, inplace=True)
            logger.info("Renamed 'mobile_number' to 'customer_id'")
        
        missing_cols = []
        if not has_month and not has_month_suffix:
            missing_cols = ["month info"]
        
        if missing_cols:
            raise ValueError(f"Missing required data: {missing_cols}")
        
        # Check for duplicate customer_id (for wide format) or customer_id + month (for long format)
        if "month" in df.columns:
            duplicates = df.duplicated(subset=["customer_id", "month"]).sum()
            if duplicates > 0:
                logger.warning(f"Found {duplicates} duplicate customer-month entries")
        else:
            # Wide format: one row per customer
            duplicates = df.duplicated(subset=["customer_id"]).sum()
            if duplicates > 0:
                logger.warning(f"Found {duplicates} duplicate customer entries")
        
        # Check for null values in key columns
        null_counts = df.isnull().sum()
        if null_counts.any():
            logger.warning(f"Null values found:\n{null_counts[null_counts > 0]}")
        
        logger.info("Data validation passed")
        return True
    
    def identify_high_value_customers(
        self, 
        df: pd.DataFrame,
        months: list = GOOD_PHASE_MONTHS,
        percentile: float = HIGH_VALUE_THRESHOLD
    ) -> pd.DataFrame:
        """Identify high-value customers based on recharge amount.
        
        High-value customers are defined as those in the 70th percentile 
        of average recharge amount in the good phase months.
        
        Supports both long format (month column) and wide format (column suffixes _6, _7, etc.)
        
        Args:
            df: Input DataFrame
            months: List of months to consider for high-value calculation
            percentile: Threshold percentile for high-value identification
            
        Returns:
            DataFrame with high_value flag added
        """
        logger.info(f"Identifying high-value customers (top {percentile*100:.0f}%)")
        
        # Check if wide format (columns like total_rech_amt_6, total_rech_amt_7)
        is_wide_format = not "month" in df.columns and any(col.endswith("_6") for col in df.columns)
        
        if is_wide_format:
            # Wide format: calculate avg recharge from columns like total_rech_amt_6
            recharge_cols = []
            for m in months:
                # Try various naming conventions
                candidates = [
                    f"total_rech_amt_{m}",
                    f"total_rech_num_{m}",
                    f"recharge_amount_{m}",
                    f"recharge_amount_m{m}",
                    f"arpu_{m}",
                ]
                for col in candidates:
                    if col in df.columns:
                        recharge_cols.append(col)
                        break
            
            if recharge_cols:
                df["avg_recharge_good_phase"] = df[recharge_cols].mean(axis=1)
            else:
                logger.warning("No recharge columns found, using ARPU or total MOU as proxy")
                # Use ARPU if available
                arpu_cols = [f"arpu_{m}" for m in months if f"arpu_{m}" in df.columns]
                if arpu_cols:
                    df["avg_recharge_good_phase"] = df[arpu_cols].mean(axis=1)
                else:
                    # Use total minutes of usage
                    mou_cols = [f"total_og_mou_{m}" for m in months if f"total_og_mou_{m}" in df.columns]
                    mou_cols += [f"total_ic_mou_{m}" for m in months if f"total_ic_mou_{m}" in df.columns]
                    if mou_cols:
                        df["avg_recharge_good_phase"] = df[mou_cols].sum(axis=1)
                    else:
                        raise ValueError("No suitable columns found for high-value identification")
        else:
            # Long format: original logic
            good_phase_data = df[df["month"].isin(months)].copy()
            
            recharge_cols = [f"recharge_amount_m{m}" for m in months 
                            if f"recharge_amount_m{m}" in good_phase_data.columns]
            
            if not recharge_cols:
                if "recharge_amount" in good_phase_data.columns:
                    avg_recharge = good_phase_data.groupby("customer_id")["recharge_amount"].mean()
                else:
                    logger.warning("No recharge amount columns found, using total usage as proxy")
                    usage_cols = [col for col in good_phase_data.columns 
                                 if "calls" in col.lower() and col != "customer_id"]
                    if usage_cols:
                        good_phase_data["total_usage"] = good_phase_data[usage_cols].sum(axis=1)
                        avg_recharge = good_phase_data.groupby("customer_id")["total_usage"].mean()
                    else:
                        raise ValueError("No suitable columns found for high-value identification")
            else:
                good_phase_data["avg_recharge"] = good_phase_data[recharge_cols].mean(axis=1)
                avg_recharge = good_phase_data.groupby("customer_id")["avg_recharge"].mean()
            
            df["avg_recharge_good_phase"] = df["customer_id"].map(avg_recharge)
        
        # Calculate threshold and identify high-value customers
        threshold = df["avg_recharge_good_phase"].quantile(percentile)
        logger.info(f"High-value threshold (P{percentile*100}): {threshold:.2f}")
        
        df["high_value"] = (df["avg_recharge_good_phase"] >= threshold).astype(int)
        
        high_value_count = df["high_value"].sum()
        logger.info(f"Identified {high_value_count} high-value customers "
                   f"({high_value_count/len(df)*100:.1f}%)")
        
        return df
    
    def define_churn(
        self, 
        df: pd.DataFrame, 
        churn_month: int = 9
    ) -> pd.DataFrame:
        """Define churn based on usage in the churn month.
        
        A customer is considered churned if they have:
        - Zero incoming calls
        - Zero outgoing calls  
        - Zero mobile internet usage
        
        Supports both long format and wide format data.
        
        Args:
            df: Input DataFrame
            churn_month: Month to use for churn definition
            
        Returns:
            DataFrame with churn flag added
        """
        logger.info(f"Defining churn for month {churn_month}")
        
        # Check if wide format
        is_wide_format = not "month" in df.columns and any(col.endswith(f"_{churn_month}") for col in df.columns)
        
        if is_wide_format:
            # Wide format: use columns like total_ic_mou_9, total_og_mou_9, vol_2g_mb_9, vol_3g_mb_9
            incoming_col = f"total_ic_mou_{churn_month}"
            outgoing_col = f"total_og_mou_{churn_month}"
            data_2g_col = f"vol_2g_mb_{churn_month}"
            data_3g_col = f"vol_3g_mb_{churn_month}"
            
            # Check which columns exist
            has_incoming = incoming_col in df.columns
            has_outgoing = outgoing_col in df.columns
            has_data_2g = data_2g_col in df.columns
            has_data_3g = data_3g_col in df.columns
            
            if has_incoming and has_outgoing:
                # Churn = zero incoming AND zero outgoing AND (no data or zero data)
                conditions = (df[incoming_col] == 0) & (df[outgoing_col] == 0)
                
                if has_data_2g or has_data_3g:
                    data_condition = pd.Series([True] * len(df))
                    if has_data_2g:
                        data_condition &= (df[data_2g_col] == 0)
                    if has_data_3g:
                        data_condition &= (df[data_3g_col] == 0)
                    conditions &= data_condition
                
                df["churned"] = conditions.astype(int)
            else:
                logger.warning(f"Missing standard call columns for month {churn_month}, using available data")
                # Try sample data column names: incoming_calls_m9, outgoing_calls_m9, data_usage_m9
                incoming_candidates = [f"incoming_calls_m{churn_month}", f"incoming_{churn_month}"]
                outgoing_candidates = [f"outgoing_calls_m{churn_month}", f"outgoing_{churn_month}"]
                data_candidates = [f"data_usage_m{churn_month}", f"data_{churn_month}"]
                
                inc_col = next((c for c in incoming_candidates if c in df.columns), None)
                out_col = next((c for c in outgoing_candidates if c in df.columns), None)
                data_col = next((c for c in data_candidates if c in df.columns), None)
                
                if inc_col and out_col:
                    conditions = (df[inc_col] == 0) & (df[out_col] == 0)
                    if data_col:
                        conditions &= (df[data_col] == 0)
                    df["churned"] = conditions.astype(int)
                else:
                    # Fallback: use any columns ending with month number
                    usage_cols = [col for col in df.columns if col.endswith(f"_{churn_month}") 
                                 and any(keyword in col.lower() for keyword in ["call", "usage", "data", "mou", "og", "ic", "vol", "recharge", "amt"])]
                    if usage_cols:
                        df["churned"] = (df[usage_cols].sum(axis=1) == 0).astype(int)
                    else:
                        raise ValueError(f"Cannot define churn: no usage columns found for month {churn_month}")
        else:
            # Long format: original logic
            churn_data = df[df["month"] == churn_month].copy()
            
            incoming_col = next((col for col in churn_data.columns 
                                if "incoming" in col.lower() and "call" in col.lower()), None)
            outgoing_col = next((col for col in churn_data.columns 
                                if "outgoing" in col.lower() and "call" in col.lower()), None)
            data_col = next((col for col in churn_data.columns 
                            if "data" in col.lower() or "internet" in col.lower() 
                            or "usage" in col.lower()), None)
            
            if incoming_col and outgoing_col and data_col:
                churn_data["churned"] = (
                    (churn_data[incoming_col] == 0) & 
                    (churn_data[outgoing_col] == 0) & 
                    (churn_data[data_col] == 0)
                ).astype(int)
            else:
                logger.warning("Using simplified churn definition - checking total usage")
                usage_cols = [col for col in churn_data.columns 
                             if any(keyword in col.lower() for keyword in ["calls", "duration", "usage"])]
                if usage_cols:
                    churn_data["total_usage"] = churn_data[usage_cols].sum(axis=1)
                    churn_data["churned"] = (churn_data["total_usage"] == 0).astype(int)
                else:
                    raise ValueError("Cannot define churn: no suitable usage columns found")
            
            # Map churn back to main DataFrame
            churn_mapping = churn_data.set_index("customer_id")["churned"].to_dict()
            df["churned"] = df["customer_id"].map(churn_mapping)
        
        churn_rate = df["churned"].mean()
        logger.info(f"Overall churn rate: {churn_rate:.2%}")
        
        return df
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        target_col: str = "churned",
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into training and testing sets.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Splitting data into train and test sets")
        
        # Remove rows with missing target
        df_clean = df.dropna(subset=[target_col])
        
        # Separate features and target
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]
        
        # Stratified split to maintain class balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Train churn rate: {y_train.mean():.2%}")
        logger.info(f"Test churn rate: {y_test.mean():.2%}")
        
        return X_train, X_test, y_train, y_test
    
    def get_high_value_subset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame to include only high-value customers.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame containing only high-value customers
        """
        if "high_value" not in df.columns:
            raise ValueError("DataFrame does not contain 'high_value' column")
        
        high_value_df = df[df["high_value"] == 1].copy()
        logger.info(f"High-value subset: {len(high_value_df)} rows")
        
        return high_value_df


def load_and_prepare_data(
    data_path: Optional[Path] = None,
    identify_high_value: bool = True,
    define_churn: bool = True,
    high_value_only: bool = True
) -> pd.DataFrame:
    """Convenience function to load and prepare data in one call.
    
    Args:
        data_path: Path to data file
        identify_high_value: Whether to identify high-value customers
        define_churn: Whether to define churn labels
        high_value_only: Whether to return only high-value customers
        
    Returns:
        Prepared DataFrame
    """
    loader = DataLoader(data_path)
    df = loader.load_data()
    loader.validate_data(df)
    
    if identify_high_value:
        df = loader.identify_high_value_customers(df)
    
    if define_churn:
        df = loader.define_churn(df)
    
    if high_value_only:
        df = loader.get_high_value_subset(df)
    
    return df
