#!/usr/bin/env python
"""Train and evaluate churn prediction models."""

import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ChurnModelTrainer
from src.config import (
    DATA_DIR, MODELS_DIR, ARTIFACTS_DIR,
    APPLY_PCA, USE_SMOTE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Train all models and save results."""
    logger.info("Starting model training...")
    
    # Load data
    X_train = pd.read_csv(DATA_DIR / "processed" / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "processed" / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "processed" / "y_train.csv").squeeze()
    y_test = pd.read_csv(DATA_DIR / "processed" / "y_test.csv").squeeze()
    
    # Load feature names from preprocessor
    feature_names = None
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    if preprocessor_path.exists():
        with open(preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)
            feature_names = preprocessor.get("feature_names")
    
    # Initialize trainer
    trainer = ChurnModelTrainer(
        apply_pca=APPLY_PCA,
        use_smote=USE_SMOTE
    )
    
    # Prepare data
    X_train_processed, X_test_processed, y_train, y_test = trainer.prepare_data(
        X_train, X_test, y_train, y_test
    )
    
    # Train models
    logger.info("Training Logistic Regression...")
    trainer.train_logistic_regression(X_train_processed, X_test_processed, y_train, y_test)
    
    logger.info("Training Random Forest...")
    trainer.train_random_forest(X_train_processed, X_test_processed, y_train, y_test)
    
    logger.info("Training Gradient Boosting...")
    trainer.train_gradient_boosting(X_train_processed, X_test_processed, y_train, y_test)
    
    # Get best model
    best_name, best_model = trainer.get_best_model(metric="test_auc")
    
    # Save models
    trainer.save_models(str(MODELS_DIR))
    
    # Save PCA model separately
    if trainer.pca_model is not None:
        with open(MODELS_DIR / "pca_model.pkl", "wb") as f:
            pickle.dump(trainer.pca_model, f)
    
    # Save model info
    model_info = {
        "model_type": best_name,
        "version": "1.0.0",
        "metrics": trainer.metrics.get(best_name, {}),
        "feature_count": X_train.shape[1],
        "pca_components": X_train_processed.shape[1] if trainer.pca_model else X_train.shape[1],
        "last_updated": datetime.now().isoformat(),
        "config": {
            "apply_pca": APPLY_PCA,
            "use_smote": USE_SMOTE
        }
    }
    
    with open(MODELS_DIR / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    # Copy best model as churn_model.pkl
    import shutil
    best_model_path = MODELS_DIR / f"{best_name}.pkl"
    churn_model_path = MODELS_DIR / "churn_model.pkl"
    if best_model_path.exists():
        shutil.copy(best_model_path, churn_model_path)
    
    # Save interpretable model (logistic regression) separately
    lr_model = trainer.models.get("logistic_regression")
    if lr_model:
        with open(MODELS_DIR / "interpretable_model.pkl", "wb") as f:
            pickle.dump(lr_model, f)
    
    # Save feature importance for best model
    if feature_names:
        try:
            importance_df = trainer.get_feature_importance(best_name, feature_names)
            importance_df.to_csv(ARTIFACTS_DIR / "feature_importance.csv", index=False)
        except Exception as e:
            logger.warning(f"Could not save feature importance: {e}")
    
    logger.info("Model training complete")
    logger.info(f"Best model: {best_name}")
    logger.info(f"Best test AUC: {trainer.metrics[best_name]['test_auc']:.4f}")


if __name__ == "__main__":
    main()
