"""Feature engineering for telecom churn prediction."""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TelecomFeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering transformer for telecom data."""
    
    def __init__(self, action_month: int = 8):
        """Initialize the feature engineer.
        
        Args:
            action_month: Month used as the prediction point (action phase)
        """
        self.action_month = action_month
        self.feature_names: List[str] = []
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer (no-op for this transformer)."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform raw data into engineered features.
        
        Args:
            X: Input DataFrame with raw features
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features")
        df = X.copy()
        
        # Create usage-based features
        df = self._create_usage_features(df)
        
        # Create recharge-based features  
        df = self._create_recharge_features(df)
        
        # Create data/internet features
        df = self._create_data_features(df)
        
        # Create trend features
        df = self._create_trend_features(df)
        
        # Create ratio features
        df = self._create_ratio_features(df)
        
        # Create interaction features
        df = self._create_interaction_features(df)
        
        # Drop original columns that were used for feature creation
        df = self._clean_dataframe(df)
        
        self.feature_names = df.columns.tolist()
        logger.info(f"Created {len(self.feature_names)} features")
        
        return df
    
    def _create_usage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create usage-based features."""
        # Total calls in action month
        call_cols_m8 = [col for col in df.columns if "call" in col.lower() and "m8" in col]
        if call_cols_m8:
            df["total_calls_m8"] = df[call_cols_m8].sum(axis=1)
        
        # Total duration in action month
        duration_cols_m8 = [col for col in df.columns 
                           if any(x in col.lower() for x in ["duration", "minutes"]) and "m8" in col]
        if duration_cols_m8:
            df["total_duration_m8"] = df[duration_cols_m8].sum(axis=1)
        
        # Average usage in good phase (months 6-7)
        for feature_type in ["calls", "duration"]:
            cols = [col for col in df.columns 
                   if feature_type in col.lower() and any(f"m{m}" in col for m in [6, 7])]
            if cols:
                df[f"avg_{feature_type}_good_phase"] = df[cols].mean(axis=1)
        
        # Usage change from good phase to action phase
        if "total_calls_m8" in df.columns and "avg_calls_good_phase" in df.columns:
            df["calls_change"] = df["total_calls_m8"] - df["avg_calls_good_phase"]
            df["calls_change_pct"] = np.where(
                df["avg_calls_good_phase"] != 0,
                (df["calls_change"] / df["avg_calls_good_phase"]) * 100,
                0
            )
        
        if "total_duration_m8" in df.columns and "avg_duration_good_phase" in df.columns:
            df["duration_change"] = df["total_duration_m8"] - df["avg_duration_good_phase"]
            df["duration_change_pct"] = np.where(
                df["avg_duration_good_phase"] != 0,
                (df["duration_change"] / df["avg_duration_good_phase"]) * 100,
                0
            )
        
        return df
    
    def _create_recharge_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create recharge-based features."""
        # Average recharge amount in good phase
        recharge_cols = [col for col in df.columns 
                        if "recharge" in col.lower() and any(f"m{m}" in col for m in [6, 7])]
        if recharge_cols:
            df["avg_recharge_good_phase"] = df[recharge_cols].mean(axis=1)
        
        # Recharge amount in action month
        recharge_m8 = [col for col in df.columns if "recharge" in col.lower() and "m8" in col]
        if recharge_m8:
            df["recharge_m8"] = df[recharge_m8[0]]
            if "avg_recharge_good_phase" in df.columns:
                df["recharge_change"] = df["recharge_m8"] - df["avg_recharge_good_phase"]
        
        # Recharge frequency features
        recharge_count_cols = [col for col in df.columns 
                              if "recharge" in col.lower() and "count" in col.lower()]
        if recharge_count_cols:
            df["avg_recharge_count"] = df[recharge_count_cols].mean(axis=1)
        
        # Days since last recharge (if available)
        last_recharge_col = next((col for col in df.columns 
                                 if "last" in col.lower() and "recharge" in col.lower()), None)
        if last_recharge_col:
            df["days_since_last_recharge"] = df[last_recharge_col]
        
        return df
    
    def _create_data_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create mobile data/internet features."""
        # Data usage in action month
        data_cols_m8 = [col for col in df.columns 
                       if any(x in col.lower() for x in ["data", "internet", "usage", "mb", "gb"]) 
                       and "m8" in col]
        if data_cols_m8:
            df["data_usage_m8"] = df[data_cols_m8].sum(axis=1)
        
        # Average data usage in good phase
        data_cols_good = [col for col in df.columns 
                         if any(x in col.lower() for x in ["data", "internet", "usage", "mb", "gb"])
                         and any(f"m{m}" in col for m in [6, 7])]
        if data_cols_good:
            df["avg_data_good_phase"] = df[data_cols_good].mean(axis=1)
        
        # Data usage change
        if "data_usage_m8" in df.columns and "avg_data_good_phase" in df.columns:
            df["data_change"] = df["data_usage_m8"] - df["avg_data_good_phase"]
            df["data_change_pct"] = np.where(
                df["avg_data_good_phase"] != 0,
                (df["data_change"] / df["avg_data_good_phase"]) * 100,
                0
            )
        
        # Data sessions (if available)
        session_cols = [col for col in df.columns if "session" in col.lower()]
        if session_cols:
            df["avg_data_sessions"] = df[session_cols].mean(axis=1)
        
        return df
    
    def _create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trend-based features across months."""
        # Calculate trend for usage metrics
        for metric in ["calls", "duration", "data"]:
            cols = sorted([col for col in df.columns 
                          if metric in col.lower() and any(f"m{m}" in col for m in [6, 7, 8])])
            if len(cols) >= 2:
                # Simple trend: difference between last and first month
                df[f"{metric}_trend"] = df[cols[-1]] - df[cols[0]]
                
                # Volatility: standard deviation across months
                df[f"{metric}_volatility"] = df[cols].std(axis=1)
                
                # Declining trend flag
                df[f"{metric}_declining"] = (df[f"{metric}_trend"] < 0).astype(int)
        
        # Overall engagement trend
        trend_cols = [col for col in df.columns if "trend" in col]
        if trend_cols:
            df["overall_trend_score"] = df[trend_cols].mean(axis=1)
        
        return df
    
    def _create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio-based features."""
        # Incoming vs Outgoing call ratio
        incoming_m8 = [col for col in df.columns if "incoming" in col.lower() and "m8" in col]
        outgoing_m8 = [col for col in df.columns if "outgoing" in col.lower() and "m8" in col]
        
        if incoming_m8 and outgoing_m8:
            incoming_sum = df[incoming_m8].sum(axis=1)
            outgoing_sum = df[outgoing_m8].sum(axis=1)
            df["incoming_outgoing_ratio"] = np.where(
                outgoing_sum != 0,
                incoming_sum / outgoing_sum,
                np.nan
            )
            df["incoming_outgoing_ratio"] = df["incoming_outgoing_ratio"].fillna(0)
        
        # On-net vs Off-net ratio (if available)
        onnet_cols = [col for col in df.columns if "onnet" in col.lower() or "on_net" in col.lower()]
        offnet_cols = [col for col in df.columns if "offnet" in col.lower() or "off_net" in col.lower()]
        
        if onnet_cols and offnet_cols:
            onnet_sum = df[onnet_cols].sum(axis=1)
            offnet_sum = df[offnet_cols].sum(axis=1)
            df["onnet_offnet_ratio"] = np.where(
                offnet_sum != 0,
                onnet_sum / offnet_sum,
                np.nan
            )
            df["onnet_offnet_ratio"] = df["onnet_offnet_ratio"].fillna(0)
        
        # Recharge to usage ratio
        if "recharge_m8" in df.columns and "total_calls_m8" in df.columns:
            df["recharge_per_call"] = np.where(
                df["total_calls_m8"] != 0,
                df["recharge_m8"] / df["total_calls_m8"],
                0
            )
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        # Engagement score
        engagement_features = []
        for col in ["total_calls_m8", "total_duration_m8", "data_usage_m8"]:
            if col in df.columns:
                engagement_features.append(col)
        
        if engagement_features:
            # Normalize features before combining
            scaler = RobustScaler()
            normalized = pd.DataFrame(
                scaler.fit_transform(df[engagement_features]),
                columns=[f"{col}_norm" for col in engagement_features],
                index=df.index
            )
            df["engagement_score"] = normalized.mean(axis=1)
        
        # Risk score (combination of declining indicators)
        declining_cols = [col for col in df.columns if "declining" in col]
        if declining_cols:
            df["risk_score"] = df[declining_cols].sum(axis=1) / len(declining_cols)
        
        # Value retention score
        if "recharge_change" in df.columns and "avg_recharge_good_phase" in df.columns:
            df["value_retention"] = np.where(
                df["avg_recharge_good_phase"] != 0,
                1 + (df["recharge_change"] / df["avg_recharge_good_phase"]),
                1
            )
        
        return df
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame by dropping unnecessary columns."""
        # Keep only engineered features and essential columns
        cols_to_drop = ["month"]  # Drop month as it's encoded in features
        
        # Keep customer_id and target if they exist
        essential_cols = ["customer_id", "churned", "high_value", "avg_recharge_good_phase"]
        
        # Drop original month-specific columns but keep engineered features
        drop_patterns = ["_m6", "_m7", "m6_", "m7_"]
        for pattern in drop_patterns:
            cols_to_drop.extend([col for col in df.columns if pattern in col])
        
        # Only drop columns that exist
        cols_to_drop = [col for col in cols_to_drop if col in df.columns 
                       and col not in essential_cols]
        
        df_clean = df.drop(columns=cols_to_drop, errors="ignore")
        
        return df_clean
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select features based on various criteria."""
    
    def __init__(
        self, 
        correlation_threshold: float = 0.95,
        variance_threshold: float = 0.01,
        missing_threshold: float = 0.5
    ):
        """Initialize feature selector.
        
        Args:
            correlation_threshold: Drop features with correlation above this
            variance_threshold: Drop features with variance below this
            missing_threshold: Drop features with missing ratio above this
        """
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.missing_threshold = missing_threshold
        self.selected_features: List[str] = []
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the selector by identifying features to keep."""
        df = X.copy()
        
        # Remove features with high missing ratio
        missing_ratio = df.isnull().mean()
        features_to_keep = missing_ratio[missing_ratio < self.missing_threshold].index.tolist()
        df = df[features_to_keep]
        
        # Remove low variance features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        variances = df[numeric_cols].var()
        high_variance_features = variances[variances > self.variance_threshold].index.tolist()
        df = df[high_variance_features]
        
        # Remove highly correlated features
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns 
                  if any(upper[column] > self.correlation_threshold)]
        
        self.selected_features = [col for col in df.columns if col not in to_drop]
        
        logger.info(f"Selected {len(self.selected_features)} features from {len(X.columns)}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by selecting features."""
        return X[self.selected_features]


def create_preprocessing_pipeline():
    """Create a complete preprocessing pipeline."""
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("feature_engineer", TelecomFeatureEngineer()),
        ("feature_selector", FeatureSelector()),
        ("scaler", RobustScaler())
    ])
    
    return pipeline
