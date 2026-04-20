#!/usr/bin/env python
"""Evaluate trained models and generate reports."""

import json
import logging
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    classification_report, roc_auc_score, average_precision_score
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_DIR, MODELS_DIR, ARTIFACTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


def plot_roc_curves(models, X_test, y_test, output_path):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Churn Prediction Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ROC curves to {output_path}")


def plot_precision_recall_curves(models, X_test, y_test, output_path):
    """Plot precision-recall curves for all models."""
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        avg_precision = average_precision_score(y_test, y_prob)
        plt.plot(recall, precision, label=f"{name} (AP = {avg_precision:.3f})")
    
    baseline = y_test.mean()
    plt.axhline(baseline, color='k', linestyle='--', label=f"Baseline ({baseline:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves - Churn Prediction Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved precision-recall curves to {output_path}")


def plot_confusion_matrix(model, X_test, y_test, output_path, model_name="Model"):
    """Plot confusion matrix."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix to {output_path}")


def generate_evaluation_report(models, X_test, y_test, metrics, output_path):
    """Generate comprehensive evaluation report."""
    report = {
        "models": {},
        "comparison": {},
        "summary": {}
    }
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        report["models"][name] = {
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "average_precision": average_precision_score(y_test, y_prob),
            "stored_metrics": metrics.get(name, {})
        }
    
    # Comparison table
    comparison = []
    for name, model_metrics in report["models"].items():
        comparison.append({
            "model": name,
            "auc": model_metrics["roc_auc"],
            "avg_precision": model_metrics["average_precision"],
            "precision": model_metrics["classification_report"]["1"]["precision"],
            "recall": model_metrics["classification_report"]["1"]["recall"],
            "f1": model_metrics["classification_report"]["1"]["f1-score"]
        })
    
    report["comparison"] = comparison
    
    # Summary - best model
    best_model = max(comparison, key=lambda x: x["auc"])
    report["summary"]["best_model"] = best_model["model"]
    report["summary"]["best_auc"] = best_model["auc"]
    report["summary"]["test_size"] = len(X_test)
    report["summary"]["churn_rate"] = float(y_test.mean())
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Saved evaluation report to {output_path}")
    return report


def main():
    """Run evaluation pipeline."""
    logger.info("Starting evaluation...")
    
    # Load data
    X_test = pd.read_csv(DATA_DIR / "processed" / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "processed" / "y_test.csv").squeeze()
    
    # Load PCA model if exists
    pca_model = None
    pca_path = MODELS_DIR / "pca_model.pkl"
    if pca_path.exists():
        with open(pca_path, "rb") as f:
            pca_model = pickle.load(f)
        X_test = pd.DataFrame(pca_model.transform(X_test), index=X_test.index)
    
    # Load models
    models = {}
    model_names = ["logistic_regression", "random_forest", "gradient_boosting"]
    
    for name in model_names:
        model_path = MODELS_DIR / f"{name}.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                models[name] = pickle.load(f)
            logger.info(f"Loaded {name}")
    
    if not models:
        raise ValueError("No models found for evaluation")
    
    # Load metrics
    metrics = {}
    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    
    # Create output directory
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    if len(models) > 1:
        plot_roc_curves(models, X_test, y_test, ARTIFACTS_DIR / "roc_curves.png")
        plot_precision_recall_curves(models, X_test, y_test, ARTIFACTS_DIR / "precision_recall_curve.png")
    
    # Plot confusion matrix for best model
    best_name = max(metrics.items(), key=lambda x: x[1].get("test_auc", 0))[0] if metrics else list(models.keys())[0]
    plot_confusion_matrix(
        models[best_name], X_test, y_test,
        ARTIFACTS_DIR / "confusion_matrix.png",
        best_name
    )
    
    # Generate report
    report = generate_evaluation_report(
        models, X_test, y_test, metrics,
        ARTIFACTS_DIR / "evaluation_report.json"
    )
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*50)
    logger.info(f"Best Model: {report['summary']['best_model']}")
    logger.info(f"Test AUC: {report['summary']['best_auc']:.4f}")
    logger.info(f"Test Samples: {report['summary']['test_size']}")
    logger.info(f"Churn Rate: {report['summary']['churn_rate']:.2%}")
    logger.info("="*50)
    
    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
