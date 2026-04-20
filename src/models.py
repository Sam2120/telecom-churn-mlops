"""Model training and evaluation for telecom churn prediction."""

import logging
from typing import Dict, List, Optional, Tuple, Any
import pickle
import json

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.config import (
    RANDOM_STATE,
    APPLY_PCA,
    PCA_VARIANCE_THRESHOLD,
    USE_SMOTE,
    SMOTE_SAMPLING_STRATEGY,
    XGBOOST_PARAMS,
    LOGISTIC_REGRESSION_PARAMS,
    RANDOM_FOREST_PARAMS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnModelTrainer:
    """Trainer class for churn prediction models."""
    
    def __init__(
        self,
        apply_pca: bool = APPLY_PCA,
        use_smote: bool = USE_SMOTE,
        experiment_name: str = "telecom_churn_prediction"
    ):
        """Initialize the model trainer.
        
        Args:
            apply_pca: Whether to apply PCA for dimensionality reduction
            use_smote: Whether to use SMOTE for class balancing
            experiment_name: MLflow experiment name
        """
        self.apply_pca = apply_pca
        self.use_smote = use_smote
        self.experiment_name = experiment_name
        self.models: Dict[str, Any] = {}
        self.pca_model: Optional[PCA] = None
        self.metrics: Dict[str, Dict] = {}
        
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        
    def prepare_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Tuple:
        """Prepare data with optional PCA and SMOTE.
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training labels
            y_test: Testing labels
            
        Returns:
            Tuple of processed (X_train, X_test, y_train, y_test)
        """
        # Apply PCA if enabled
        if self.apply_pca:
            logger.info(f"Applying PCA (variance threshold: {PCA_VARIANCE_THRESHOLD})")
            self.pca_model = PCA(n_components=PCA_VARIANCE_THRESHOLD, random_state=RANDOM_STATE)
            X_train = pd.DataFrame(
                self.pca_model.fit_transform(X_train),
                index=X_train.index
            )
            X_test = pd.DataFrame(
                self.pca_model.transform(X_test),
                index=X_test.index
            )
            logger.info(f"PCA reduced dimensions to {X_train.shape[1]} components")
        
        return X_train, X_test, y_train, y_test
    
    def create_pipeline(self, base_model) -> ImbPipeline:
        """Create a pipeline with optional SMOTE.
        
        Args:
            base_model: Base classifier model
            
        Returns:
            Imbalanced-learn Pipeline
        """
        steps = []
        
        if self.use_smote:
            steps.append(("smote", SMOTE(
                sampling_strategy=SMOTE_SAMPLING_STRATEGY,
                random_state=RANDOM_STATE
            )))
        
        steps.append(("classifier", base_model))
        
        return ImbPipeline(steps)
    
    def train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> LogisticRegression:
        """Train and evaluate logistic regression model.
        
        Returns:
            Trained LogisticRegression model
        """
        model_name = "logistic_regression"
        logger.info(f"Training {model_name}")
        
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            mlflow.log_params(LOGISTIC_REGRESSION_PARAMS)
            mlflow.log_param("use_smote", self.use_smote)
            mlflow.log_param("apply_pca", self.apply_pca)
            
            # Create and train model
            base_model = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
            pipeline = self.create_pipeline(base_model)
            pipeline.fit(X_train, y_train)
            
            # Get the classifier from pipeline
            model = pipeline.named_steps["classifier"]
            self.models[model_name] = pipeline
            
            # Evaluate
            metrics = self.evaluate_model(pipeline, X_train, X_test, y_train, y_test)
            self.metrics[model_name] = metrics
            
            # Log metrics
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            
            # Log model
            mlflow.sklearn.log_model(pipeline, model_name)
            
            logger.info(f"{model_name} - AUC: {metrics['test_auc']:.4f}")
            
        return model
    
    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> RandomForestClassifier:
        """Train and evaluate random forest model.
        
        Returns:
            Trained RandomForestClassifier model
        """
        model_name = "random_forest"
        logger.info(f"Training {model_name}")
        
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            mlflow.log_params(RANDOM_FOREST_PARAMS)
            mlflow.log_param("use_smote", self.use_smote)
            mlflow.log_param("apply_pca", self.apply_pca)
            
            # Create and train model
            base_model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
            pipeline = self.create_pipeline(base_model)
            pipeline.fit(X_train, y_train)
            
            model = pipeline.named_steps["classifier"]
            self.models[model_name] = pipeline
            
            # Evaluate
            metrics = self.evaluate_model(pipeline, X_train, X_test, y_train, y_test)
            self.metrics[model_name] = metrics
            
            # Log metrics
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            
            # Log feature importances
            importances = model.feature_importances_
            feature_importance = dict(zip(
                [f"feature_{i}" for i in range(len(importances))],
                importances
            ))
            mlflow.log_dict(feature_importance, "feature_importances.json")
            
            # Log model
            mlflow.sklearn.log_model(pipeline, model_name)
            
            logger.info(f"{model_name} - AUC: {metrics['test_auc']:.4f}")
            
        return model
    
    def train_gradient_boosting(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> GradientBoostingClassifier:
        """Train and evaluate gradient boosting model.
        
        Returns:
            Trained GradientBoostingClassifier model
        """
        model_name = "gradient_boosting"
        logger.info(f"Training {model_name}")
        
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            gb_params = {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "random_state": RANDOM_STATE
            }
            mlflow.log_params(gb_params)
            mlflow.log_param("use_smote", self.use_smote)
            mlflow.log_param("apply_pca", self.apply_pca)
            
            # Create and train model
            base_model = GradientBoostingClassifier(**gb_params)
            pipeline = self.create_pipeline(base_model)
            pipeline.fit(X_train, y_train)
            
            model = pipeline.named_steps["classifier"]
            self.models[model_name] = pipeline
            
            # Evaluate
            metrics = self.evaluate_model(pipeline, X_train, X_test, y_train, y_test)
            self.metrics[model_name] = metrics
            
            # Log metrics
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            
            # Log model
            mlflow.sklearn.log_model(pipeline, model_name)
            
            logger.info(f"{model_name} - AUC: {metrics['test_auc']:.4f}")
            
        return model
    
    def evaluate_model(
        self,
        model,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict:
        """Evaluate model performance.
        
        Args:
            model: Trained model
            X_train: Training features
            X_test: Testing features
            y_train: Training labels
            y_test: Testing labels
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_train_prob = model.predict_proba(X_train)[:, 1]
        y_test_prob = model.predict_proba(X_test)[:, 1]
        
        # Training metrics
        metrics["train_accuracy"] = accuracy_score(y_train, y_train_pred)
        metrics["train_precision"] = precision_score(y_train, y_train_pred, zero_division=0)
        metrics["train_recall"] = recall_score(y_train, y_train_pred, zero_division=0)
        metrics["train_f1"] = f1_score(y_train, y_train_pred, zero_division=0)
        metrics["train_auc"] = roc_auc_score(y_train, y_train_prob)
        
        # Testing metrics
        metrics["test_accuracy"] = accuracy_score(y_test, y_test_pred)
        metrics["test_precision"] = precision_score(y_test, y_test_pred, zero_division=0)
        metrics["test_recall"] = recall_score(y_test, y_test_pred, zero_division=0)
        metrics["test_f1"] = f1_score(y_test, y_test_pred, zero_division=0)
        metrics["test_auc"] = roc_auc_score(y_test, y_test_prob)
        metrics["test_average_precision"] = average_precision_score(y_test, y_test_prob)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        metrics["confusion_matrix"] = cm.tolist()
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            scoring="roc_auc"
        )
        metrics["cv_auc_mean"] = cv_scores.mean()
        metrics["cv_auc_std"] = cv_scores.std()
        
        return metrics
    
    def get_feature_importance(
        self,
        model_name: str = "random_forest",
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get feature importance from a trained model.
        
        Args:
            model_name: Name of the model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importances
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train it first.")
        
        model = self.models[model_name].named_steps["classifier"]
        
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            raise ValueError(f"Model {model_name} doesn't have feature importance")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)
        
        return importance_df
    
    def save_models(self, output_dir: str):
        """Save all trained models to disk.
        
        Args:
            output_dir: Directory to save models
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, model in self.models.items():
            filepath = os.path.join(output_dir, f"{name}.pkl")
            with open(filepath, "wb") as f:
                pickle.dump(model, f)
            logger.info(f"Saved {name} to {filepath}")
        
        # Save PCA model if exists
        if self.pca_model is not None:
            pca_path = os.path.join(output_dir, "pca_model.pkl")
            with open(pca_path, "wb") as f:
                pickle.dump(self.pca_model, f)
            logger.info(f"Saved PCA model to {pca_path}")
        
        # Save metrics
        metrics_path = os.path.join(output_dir, "metrics.json")
        # Convert numpy types to native Python types for JSON serialization
        serializable_metrics = {}
        for model, model_metrics in self.metrics.items():
            serializable_metrics[model] = {
                k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in model_metrics.items()
                if not isinstance(v, (list, np.ndarray)) or k == "confusion_matrix"
            }
        
        with open(metrics_path, "w") as f:
            json.dump(serializable_metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
    
    def get_best_model(self, metric: str = "test_auc") -> Tuple[str, Any]:
        """Get the best performing model based on a metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model_name, model_object)
        """
        best_model = None
        best_score = -np.inf
        
        for name, metrics in self.metrics.items():
            if metric in metrics and metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model = name
        
        if best_model is None:
            raise ValueError(f"No models found or metric {metric} not available")
        
        logger.info(f"Best model: {best_model} ({metric}: {best_score:.4f})")
        return best_model, self.models[best_model]


class ModelExplainer:
    """Generate explanations for model predictions."""
    
    def __init__(self, model, feature_names: Optional[List[str]] = None):
        """Initialize explainer.
        
        Args:
            model: Trained model
            feature_names: Names of features
        """
        self.model = model
        self.feature_names = feature_names
        
    def explain_prediction(self, X: pd.DataFrame, customer_id: Optional[str] = None) -> Dict:
        """Generate explanation for a single prediction.
        
        Args:
            X: Feature values (single row)
            customer_id: Optional customer identifier
            
        Returns:
            Explanation dictionary
        """
        if isinstance(X, pd.DataFrame):
            X = X.iloc[0:1]  # Take first row
        
        # Get prediction probability
        prob = self.model.predict_proba(X)[0, 1]
        prediction = 1 if prob > 0.5 else 0
        
        # Get feature importance contribution
        explanation = {
            "customer_id": customer_id,
            "churn_probability": float(prob),
            "prediction": "Churn" if prediction == 1 else "No Churn",
            "risk_level": self._get_risk_level(prob),
        }
        
        # Add feature contributions if available
        if hasattr(self.model.named_steps["classifier"], "coef_"):
            coef = self.model.named_steps["classifier"].coef_[0]
            contributions = X.values[0] * coef
            
            if self.feature_names:
                feature_contrib = list(zip(self.feature_names, contributions))
                feature_contrib.sort(key=lambda x: abs(x[1]), reverse=True)
                explanation["top_factors"] = [
                    {"feature": f, "contribution": float(c)}
                    for f, c in feature_contrib[:5]
                ]
        
        return explanation
    
    def _get_risk_level(self, probability: float) -> str:
        """Convert probability to risk level."""
        if probability >= 0.8:
            return "Very High"
        elif probability >= 0.6:
            return "High"
        elif probability >= 0.4:
            return "Medium"
        elif probability >= 0.2:
            return "Low"
        else:
            return "Very Low"


def train_all_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    apply_pca: bool = True,
    use_smote: bool = True
) -> ChurnModelTrainer:
    """Convenience function to train all models.
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
        apply_pca: Whether to apply PCA
        use_smote: Whether to use SMOTE
        
    Returns:
        ChurnModelTrainer with all trained models
    """
    trainer = ChurnModelTrainer(apply_pca=apply_pca, use_smote=use_smote)
    
    # Prepare data
    X_train_processed, X_test_processed, y_train, y_test = trainer.prepare_data(
        X_train, X_test, y_train, y_test
    )
    
    # Train models
    trainer.train_logistic_regression(X_train_processed, X_test_processed, y_train, y_test)
    trainer.train_random_forest(X_train_processed, X_test_processed, y_train, y_test)
    trainer.train_gradient_boosting(X_train_processed, X_test_processed, y_train, y_test)
    
    return trainer
