"""MLflow utilities for experiment tracking and model registry."""

import logging
import os
from typing import Optional, Dict, Any

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_mlflow():
    """Set up MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")


class MLflowModelRegistry:
    """Handle MLflow model registry operations."""
    
    def __init__(self, tracking_uri: Optional[str] = None):
        """Initialize the model registry.
        
        Args:
            tracking_uri: MLflow tracking URI
        """
        self.tracking_uri = tracking_uri or MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient()
        
    def register_model(
        self,
        model_name: str,
        run_id: str,
        model_artifact_path: str = "model",
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> str:
        """Register a model to MLflow Model Registry.
        
        Args:
            model_name: Name for the registered model
            run_id: MLflow run ID
            model_artifact_path: Path to model artifact
            tags: Optional tags for the model version
            description: Optional description
            
        Returns:
            Model version string
        """
        model_uri = f"runs:/{run_id}/{model_artifact_path}"
        
        # Register model
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
        
        # Add description if provided
        if description:
            self.client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=description
            )
        
        # Add tags if provided
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(
                    name=model_name,
                    version=model_version.version,
                    key=key,
                    value=value
                )
        
        logger.info(f"Registered model {model_name} version {model_version.version}")
        return model_version.version
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing: bool = True
    ):
        """Transition model to a different stage.
        
        Args:
            model_name: Registered model name
            version: Model version
            stage: Target stage (Staging, Production, Archived)
            archive_existing: Whether to archive existing versions in target stage
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing
        )
        logger.info(f"Transitioned {model_name} v{version} to {stage}")
    
    def get_production_model(self, model_name: str) -> Optional[str]:
        """Get the latest production model URI.
        
        Args:
            model_name: Registered model name
            
        Returns:
            Model URI or None
        """
        try:
            versions = self.client.get_latest_versions(model_name, stages=["Production"])
            if versions:
                version = versions[0]
                return f"models:/{model_name}/{version.version}"
            return None
        except Exception as e:
            logger.error(f"Error getting production model: {e}")
            return None
    
    def compare_models(
        self,
        model_name: str,
        metric: str = "test_auc"
    ) -> Dict[str, Any]:
        """Compare all versions of a model.
        
        Args:
            model_name: Registered model name
            metric: Metric to compare
            
        Returns:
            Dictionary with comparison results
        """
        versions = self.client.search_model_versions(f"name='{model_name}'")
        
        results = []
        for version in versions:
            run = self.client.get_run(version.run_id)
            metric_value = run.data.metrics.get(metric, 0)
            results.append({
                "version": version.version,
                "stage": version.current_stage,
                "status": version.status,
                metric: metric_value,
                "run_id": version.run_id
            })
        
        # Sort by metric value
        results.sort(key=lambda x: x[metric], reverse=True)
        
        return {
            "model_name": model_name,
            "metric": metric,
            "versions": results,
            "best_version": results[0] if results else None
        }
    
    def promote_best_model(
        self,
        model_name: str,
        metric: str = "test_auc",
        threshold: float = 0.8
    ) -> bool:
        """Automatically promote the best model to production.
        
        Args:
            model_name: Registered model name
            metric: Metric to evaluate
            threshold: Minimum metric value for promotion
            
        Returns:
            True if promotion successful
        """
        comparison = self.compare_models(model_name, metric)
        
        best = comparison["best_version"]
        if best and best[metric] >= threshold:
            self.transition_model_stage(
                model_name,
                best["version"],
                "Production",
                archive_existing=True
            )
            logger.info(
                f"Promoted {model_name} v{best['version']} to Production "
                f"({metric}: {best[metric]:.4f})"
            )
            return True
        else:
            logger.info(
                f"No model met promotion threshold ({threshold}) for {metric}"
            )
            return False


def log_model_with_signature(
    model,
    artifact_path: str,
    X_sample: Any,
    y_sample: Optional[Any] = None,
    params: Optional[Dict] = None,
    metrics: Optional[Dict] = None,
    tags: Optional[Dict] = None
) -> str:
    """Log a model with signature to MLflow.
    
    Args:
        model: Trained model
        artifact_path: Path for model artifact
        X_sample: Sample input for signature inference
        y_sample: Optional sample output
        params: Optional parameters to log
        metrics: Optional metrics to log
        tags: Optional tags to log
        
    Returns:
        Run ID
    """
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
        
        # Log parameters
        if params:
            mlflow.log_params(params)
        
        # Log metrics
        if metrics:
            mlflow.log_metrics(metrics)
        
        # Log tags
        if tags:
            mlflow.set_tags(tags)
        
        # Log model with signature
        mlflow.sklearn.log_model(
            model,
            artifact_path=artifact_path,
            signature=mlflow.models.infer_signature(X_sample, y_sample),
            input_example=X_sample.iloc[:5] if hasattr(X_sample, 'iloc') else X_sample[:5]
        )
        
        logger.info(f"Logged model to run {run_id}")
        return run_id


def get_run_history(experiment_name: Optional[str] = None) -> Dict:
    """Get run history for an experiment.
    
    Args:
        experiment_name: Name of experiment (uses default if None)
        
    Returns:
        Dictionary with run history
    """
    experiment_name = experiment_name or MLFLOW_EXPERIMENT_NAME
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if not experiment:
        return {"experiment": experiment_name, "runs": []}
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )
    
    return {
        "experiment": experiment_name,
        "experiment_id": experiment.experiment_id,
        "num_runs": len(runs),
        "runs": runs.to_dict(orient="records")
    }
