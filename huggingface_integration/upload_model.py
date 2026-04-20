"""Upload trained models to Hugging Face Hub."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
from huggingface_hub.repocard import ModelCard, ModelCardData

from src.config import MODELS_DIR, HF_TOKEN, HF_MODEL_REPO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HuggingFaceUploader:
    """Handle uploading models to Hugging Face Hub."""
    
    def __init__(
        self,
        token: Optional[str] = None,
        repo_id: Optional[str] = None
    ):
        """Initialize uploader.
        
        Args:
            token: Hugging Face API token
            repo_id: Repository ID (username/repo-name)
        """
        self.token = token or HF_TOKEN or os.getenv("HF_TOKEN")
        self.repo_id = repo_id or HF_MODEL_REPO
        self.api = HfApi(token=self.token)
        
        if not self.token:
            raise ValueError("Hugging Face token required")
        
    def create_model_repository(
        self,
        private: bool = False,
        exist_ok: bool = True
    ):
        """Create a model repository on Hugging Face.
        
        Args:
            private: Whether repository should be private
            exist_ok: Don't error if repo already exists
        """
        try:
            create_repo(
                repo_id=self.repo_id,
                token=self.token,
                private=private,
                exist_ok=exist_ok,
                repo_type="model"
            )
            logger.info(f"Created/verified repository: {self.repo_id}")
        except Exception as e:
            logger.error(f"Error creating repository: {e}")
            raise
    
    def create_model_card(
        self,
        model_name: str = "Telecom Churn Prediction Model",
        description: Optional[str] = None,
        metrics: Optional[Dict] = None
    ) -> ModelCard:
        """Create a model card for the repository.
        
        Args:
            model_name: Name of the model
            description: Model description
            metrics: Model performance metrics
            
        Returns:
            ModelCard object
        """
        card_data = ModelCardData(
            language="en",
            license="mit",
            library_name="scikit-learn",
            tags=["tabular-classification", "churn-prediction", "telecom", "mlops"],
            datasets=["telecom-customer-data"],
            metrics=["auc", "f1", "precision", "recall"],
            model_name=model_name,
        )
        
        description = description or """
# Telecom High-Value Customer Churn Prediction

This model predicts churn probability for high-value customers in the telecom industry.

## Model Description

- **Task:** Binary Classification (Churn vs. No Churn)
- **Target:** High-value prepaid customers (top 70th percentile by recharge amount)
- **Features:** Customer usage patterns, recharge behavior, call duration, data usage
- **Training Data:** 4 months of customer behavior data (June-September)
- **Model Type:** Ensemble of Random Forest, Gradient Boosting, and Logistic Regression
- **Class Imbalance Handling:** SMOTE + Class Weights
- **Dimensionality Reduction:** PCA (95% variance retention)

## Intended Use

This model is designed for telecom operators to:
- Identify high-value customers at risk of churning
- Enable proactive retention campaigns
- Optimize resource allocation for customer retention

## Performance
"""
        
        if metrics:
            description += "\n\n### Test Set Metrics\n"
            for metric, value in metrics.items():
                description += f"- **{metric}:** {value:.4f}\n"
        
        description += """
## Training Data

- **Time Period:** 4 consecutive months (June-September)
- **Geography:** Multi-region telecom operator data
- **Customer Segment:** High-value prepaid customers
- **Churn Definition:** Zero usage (calls + data) in the churn month

## Limitations

- Model trained on specific telecom operator data; may not generalize to other regions
- Churn prediction window is 1 month ahead
- High-value customer definition based on recharge amount percentile
- Does not account for external factors (network quality, competitor offers)

## Ethical Considerations

- Model predictions should not be the sole basis for customer treatment
- Privacy: Ensure compliance with data protection regulations
- Fairness: Monitor for bias across customer segments

## Citation

```bibtex
@software{telecom_churn_mlops,
  title = {MLOps-Driven End-to-End Pipeline for Telecom Churn Prediction},
  author = {MLOps Team},
  year = {2024}
}
```
"""
        
        card = ModelCard.from_template(
            card_data=card_data,
            template_path=None,
            content=description
        )
        
        return card
    
    def upload_model_files(
        self,
        model_dir: Path = MODELS_DIR,
        commit_message: Optional[str] = None
    ):
        """Upload model files to Hugging Face.
        
        Args:
            model_dir: Directory containing model files
            commit_message: Git commit message
        """
        commit_message = commit_message or "Update model files"
        
        # Upload all model files
        for file_path in model_dir.glob("*.pkl"):
            logger.info(f"Uploading {file_path.name}")
            upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_path.name,
                repo_id=self.repo_id,
                token=self.token,
                commit_message=f"{commit_message}: {file_path.name}"
            )
        
        # Upload metrics
        metrics_path = model_dir / "metrics.json"
        if metrics_path.exists():
            upload_file(
                path_or_fileobj=str(metrics_path),
                path_in_repo="metrics.json",
                repo_id=self.repo_id,
                token=self.token,
                commit_message="Update metrics"
            )
    
    def upload_model_card(self, card: ModelCard):
        """Upload model card to repository.
        
        Args:
            card: ModelCard object
        """
        card.save("README.md")
        upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=self.repo_id,
            token=self.token,
            commit_message="Update model card"
        )
        logger.info("Uploaded model card")
    
    def upload_complete_model(
        self,
        model_dir: Path = MODELS_DIR,
        metrics: Optional[Dict] = None,
        version: str = "1.0.0"
    ):
        """Complete model upload workflow.
        
        Args:
            model_dir: Directory with model files
            metrics: Model performance metrics
            version: Model version
        """
        logger.info(f"Starting upload to {self.repo_id}")
        
        # Create repository
        self.create_model_repository()
        
        # Upload model files
        self.upload_model_files(
            model_dir=model_dir,
            commit_message=f"Model version {version}"
        )
        
        # Create and upload model card
        card = self.create_model_card(metrics=metrics)
        self.upload_model_card(card)
        
        logger.info("Model upload complete")
    
    def upload_to_model_hub(
        self,
        model_path: Path,
        model_name: str,
        tags: Optional[list] = None
    ):
        """Upload as a Hugging Face Model Hub model.
        
        Args:
            model_path: Path to model file
            model_name: Name for the model
            tags: Tags for the model
        """
        from transformers import AutoModel, AutoConfig
        
        # Create model card data
        card_data = ModelCardData(
            language="en",
            license="mit",
            library_name="sklearn",
            tags=tags or ["tabular-classification", "churn-prediction"],
        )
        
        # For sklearn models, we create a simple wrapper
        # Note: This is a simplified approach; for production, consider using skops
        logger.info(f"Uploading {model_name} to Model Hub")
        
        # Upload the pickled model
        upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=f"{model_name}.pkl",
            repo_id=self.repo_id,
            token=self.token,
        )


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face")
    parser.add_argument("--repo", help="Repository ID (username/repo-name)")
    parser.add_argument("--model-dir", default=str(MODELS_DIR), help="Model directory")
    parser.add_argument("--version", default="1.0.0", help="Model version")
    
    args = parser.parse_args()
    
    # Load metrics if available
    metrics = {}
    metrics_path = Path(args.model_dir) / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            all_metrics = json.load(f)
            # Get best model metrics
            best_model = max(all_metrics.items(), key=lambda x: x[1].get("test_auc", 0))
            metrics = {
                "auc": best_model[1].get("test_auc", 0),
                "f1": best_model[1].get("test_f1", 0),
                "precision": best_model[1].get("test_precision", 0),
                "recall": best_model[1].get("test_recall", 0),
            }
    
    uploader = HuggingFaceUploader(repo_id=args.repo)
    uploader.upload_complete_model(
        model_dir=Path(args.model_dir),
        metrics=metrics,
        version=args.version
    )


if __name__ == "__main__":
    main()
