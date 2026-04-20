"""Download models from Hugging Face Hub."""

import logging
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, snapshot_download

from src.config import MODELS_DIR, HF_MODEL_REPO, HF_TOKEN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HuggingFaceDownloader:
    """Handle downloading models from Hugging Face Hub."""
    
    def __init__(
        self,
        token: Optional[str] = None,
        repo_id: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """Initialize downloader.
        
        Args:
            token: Hugging Face API token
            repo_id: Repository ID
            cache_dir: Cache directory for downloads
        """
        self.token = token or HF_TOKEN or os.getenv("HF_TOKEN")
        self.repo_id = repo_id or HF_MODEL_REPO
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface")
        
    def download_model_file(
        self,
        filename: str,
        local_dir: Path = MODELS_DIR,
        revision: Optional[str] = None
    ) -> Path:
        """Download a specific model file.
        
        Args:
            filename: Name of file to download
            local_dir: Local directory to save to
            revision: Git revision to download from
            
        Returns:
            Path to downloaded file
        """
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading {filename} from {self.repo_id}")
        
        downloaded_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=filename,
            revision=revision,
            cache_dir=self.cache_dir,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            token=self.token
        )
        
        logger.info(f"Downloaded to {downloaded_path}")
        return Path(downloaded_path)
    
    def download_all_models(
        self,
        local_dir: Path = MODELS_DIR,
        revision: Optional[str] = None,
        ignore_patterns: Optional[list] = None
    ) -> Path:
        """Download all model files from repository.
        
        Args:
            local_dir: Local directory to save to
            revision: Git revision
            ignore_patterns: Patterns to ignore
            
        Returns:
            Path to download directory
        """
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading all files from {self.repo_id}")
        
        downloaded_path = snapshot_download(
            repo_id=self.repo_id,
            revision=revision,
            cache_dir=self.cache_dir,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            token=self.token,
            ignore_patterns=ignore_patterns or ["*.md", "*.txt", ".gitattributes"]
        )
        
        logger.info(f"Downloaded to {downloaded_path}")
        return Path(downloaded_path)
    
    def download_model_by_version(
        self,
        version: str,
        local_dir: Path = MODELS_DIR
    ) -> Path:
        """Download a specific model version.
        
        Args:
            version: Version tag (e.g., "v1.0.0")
            local_dir: Local directory
            
        Returns:
            Path to downloaded model
        """
        return self.download_all_models(
            local_dir=local_dir,
            revision=f"refs/tags/{version}"
        )
    
    def download_metrics(self, local_dir: Path = MODELS_DIR) -> Optional[dict]:
        """Download and load metrics file.
        
        Args:
            local_dir: Local directory
            
        Returns:
            Metrics dictionary or None
        """
        import json
        
        try:
            metrics_path = self.download_model_file("metrics.json", local_dir)
            with open(metrics_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not download metrics: {e}")
            return None
    
    def get_model_info(self) -> dict:
        """Get information about the model repository.
        
        Returns:
            Repository information
        """
        from huggingface_hub import model_info
        
        info = model_info(self.repo_id, token=self.token)
        return {
            "id": info.id,
            "sha": info.sha,
            "tags": info.tags,
            "pipeline_tag": info.pipeline_tag,
            "downloads": info.downloads,
            "likes": info.likes,
            "card": info.card_data if hasattr(info, "card_data") else None
        }


def download_for_deployment(
    repo_id: Optional[str] = None,
    output_dir: Path = MODELS_DIR
) -> bool:
    """Download models for deployment.
    
    Args:
        repo_id: Repository ID (uses config default if None)
        output_dir: Output directory
        
    Returns:
        True if successful
    """
    try:
        downloader = HuggingFaceDownloader(repo_id=repo_id)
        downloader.download_all_models(local_dir=output_dir)
        
        # Also download metrics
        metrics = downloader.download_metrics(output_dir)
        if metrics:
            logger.info(f"Model metrics: {metrics}")
        
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download model from Hugging Face")
    parser.add_argument("--repo", help="Repository ID")
    parser.add_argument("--output", default=str(MODELS_DIR), help="Output directory")
    parser.add_argument("--version", help="Specific version to download")
    parser.add_argument("--file", help="Specific file to download")
    
    args = parser.parse_args()
    
    downloader = HuggingFaceDownloader(repo_id=args.repo)
    
    if args.file:
        downloader.download_model_file(args.file, Path(args.output))
    elif args.version:
        downloader.download_model_by_version(args.version, Path(args.output))
    else:
        downloader.download_all_models(local_dir=Path(args.output))
        metrics = downloader.download_metrics(Path(args.output))
        if metrics:
            print(f"\nModel Metrics:")
            print(f"AUC: {metrics.get('auc', 'N/A')}")
            print(f"F1: {metrics.get('f1', 'N/A')}")
            print(f"Precision: {metrics.get('precision', 'N/A')}")
            print(f"Recall: {metrics.get('recall', 'N/A')}")


if __name__ == "__main__":
    main()
