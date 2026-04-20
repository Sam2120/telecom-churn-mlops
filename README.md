# MLOps-Driven End-to-End Pipeline for Telecom Churn Prediction

[![CI/CD](https://github.com/your-org/telecom-churn-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/your-org/telecom-churn-mlops/actions)
[![DVC](https://img.shields.io/badge/DVC-enabled-blue)](https://dvc.org)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-orange)](https://mlflow.org)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/your-model)

A production-ready MLOps pipeline for predicting high-value customer churn in the telecom industry. This project demonstrates end-to-end machine learning operations with automated data versioning, experiment tracking, CI/CD deployment, and REST API serving.

![Architecture Diagram](docs/assets/architecture.png)

## Project Overview

### Problem Statement

In the prepaid telecom market, customers can silently discontinue usage without formally notifying the operator, making churn difficult to detect and prevent. With annual churn rates ranging from 15-25% and customer acquisition costing 5-10x more than retention, telecom companies face significant revenue leakage from high-value customers.

This project predicts churn among high-value prepaid customers using 4 months of customer-level data (June-September), enabling proactive retention strategies.

### Key Features

- **High-Value Customer Focus**: Identifies top 70th percentile customers by recharge amount
- **Usage-Based Churn Definition**: Zero calls (incoming/outgoing) + zero data usage in churn month
- **MLOps Integration**: Complete ML lifecycle with DVC, MLflow, CI/CD, and Docker
- **Multi-Model Ensemble**: Random Forest, Gradient Boosting, and Logistic Regression
- **Interpretability**: Feature importance and prediction explanations
- **REST API**: FastAPI deployment for real-time predictions
- **Hugging Face**: Model sharing and interactive demo

## Repository Structure

```
telecom-churn-mlops/
├── .dvc/                       # DVC configuration
├── .github/workflows/           # CI/CD pipelines
│   ├── ci-cd.yml               # Main CI/CD workflow
│   └── dvc-sync.yml            # Automated data sync
├── api/                        # FastAPI application
│   └── main.py                 # REST API endpoints
├── configs/                    # Configuration files
├── data/                       # Data directory (DVC managed)
│   ├── raw/                    # Raw input data
│   ├── interim/                # Intermediate processed data
│   └── processed/              # Final features and splits
├── huggingface_integration/    # Hugging Face utilities
│   ├── upload_model.py         # Model upload script
│   ├── download_model.py     # Model download script
│   └── spaces_app.py         # Gradio demo app
├── models/                     # Trained models (DVC managed)
├── notebooks/                  # Jupyter notebooks for exploration
├── scripts/                    # Pipeline scripts
│   ├── load_data.py           # Data loading
│   ├── preprocess.py          # Feature engineering
│   ├── split_data.py          # Train/test split
│   ├── train_models.py        # Model training
│   └── evaluate.py            # Model evaluation
├── src/                        # Source code
│   ├── config.py              # Configuration settings
│   ├── data_loader.py         # Data loading utilities
│   ├── feature_engineering.py # Feature engineering
│   ├── models.py              # Model training
│   └── mlflow_utils.py        # MLflow integration
├── tests/                      # Unit tests
├── Dockerfile                  # API container
├── Dockerfile.jupyter          # Jupyter container
├── docker-compose.yml          # Local stack orchestration
├── dvc.yaml                    # DVC pipeline definition
└── requirements.txt            # Python dependencies
```

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- AWS CLI (for S3/DVC)
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-org/telecom-churn-mlops.git
cd telecom-churn-mlops
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your credentials
```

4. **Set up DVC and pull data**
```bash
dvc remote modify s3-remote --local access_key_id $AWS_ACCESS_KEY_ID
dvc remote modify s3-remote --local secret_access_key $AWS_SECRET_ACCESS_KEY
dvc pull
```

### Running the Pipeline

#### Option 1: DVC Pipeline (Recommended)
```bash
# Run complete pipeline
dvc repro

# Visualize pipeline
dvc dag
```

#### Option 2: Individual Scripts
```bash
# Step-by-step execution
python scripts/load_data.py
python scripts/preprocess.py
python scripts/split_data.py
python scripts/train_models.py
python scripts/evaluate.py
```

#### Option 3: Docker Compose
```bash
# Start full stack (MLflow + API)
docker-compose up -d

# Access services
# API: http://localhost:8000
# MLflow: http://localhost:5000

# View logs
docker-compose logs -f api
```

## API Usage

### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/model/info` | GET | Model information |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/predict/file` | POST | CSV file predictions |
| `/explain` | POST | Prediction explanation |

### Example Request

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST001",
    "recharge_amount_m6": 500,
    "recharge_amount_m7": 450,
    "recharge_amount_m8": 300,
    "total_calls_m6": 150,
    "total_calls_m7": 140,
    "total_calls_m8": 80,
    "total_duration_m6": 3000,
    "total_duration_m7": 2800,
    "total_duration_m8": 1500,
    "incoming_calls_m6": 50,
    "incoming_calls_m7": 45,
    "incoming_calls_m8": 20,
    "outgoing_calls_m6": 100,
    "outgoing_calls_m7": 95,
    "outgoing_calls_m8": 60,
    "data_usage_m6": 2048,
    "data_usage_m7": 1800,
    "data_usage_m8": 512
  }'
```

### Example Response

```json
{
  "customer_id": "CUST001",
  "churn_probability": 0.82,
  "prediction": "Churn",
  "risk_level": "Very High",
  "confidence": "Very High"
}
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {"customer_id": "C001", "recharge_amount_m6": 500, ...},
      {"customer_id": "C002", "recharge_amount_m6": 300, ...}
    ]
  }'
```

## MLOps Components

### 1. Data Versioning (DVC)

```bash
# Track data changes
dvc add data/raw/telecom_data.csv
git add data/raw/telecom_data.csv.dvc
git commit -m "Add new data version"

# Push to remote storage
dvc push

# Pull specific version
dvc pull
```

### 2. Experiment Tracking (MLflow)

```bash
# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# View at http://localhost:5000
```

Tracked automatically:
- Model parameters
- Performance metrics (AUC, F1, Precision, Recall)
- Artifacts (models, plots)
- Model versions

### 3. CI/CD Pipeline (GitHub Actions)

Workflow stages:
1. **Test**: Linting, unit tests, coverage
2. **Data**: DVC pipeline execution
3. **Train**: Model training with MLflow tracking
4. **Build**: Docker image creation
5. **Deploy**: Staging → Production deployment
6. **Hugging Face**: Model hub upload

### 4. Model Registry

```python
from src.mlflow_utils import MLflowModelRegistry

registry = MLflowModelRegistry()

# Register model
version = registry.register_model(
    model_name="telecom_churn",
    run_id="...",
    tags={"stage": "production", "auc": "0.85"}
)

# Promote to production
registry.transition_model_stage("telecom_churn", version, "Production")

# Get production model
model_uri = registry.get_production_model("telecom_churn")
```

## Model Details

### Feature Engineering

- **Usage Features**: Call counts, duration (incoming/outgoing)
- **Recharge Features**: Amount, frequency, trends
- **Data Features**: Internet usage, sessions
- **Trend Features**: Month-over-month changes
- **Ratio Features**: Incoming/outgoing ratio, on-net/off-net
- **Interaction Features**: Engagement score, risk score

### Models Trained

| Model | Type | Use Case |
|-------|------|----------|
| Logistic Regression | Linear | Interpretability |
| Random Forest | Ensemble | Best performance |
| Gradient Boosting | Ensemble | Accuracy |

### Class Imbalance Handling

- **SMOTE**: Synthetic minority oversampling
- **Class Weights**: Balanced weighting

### Dimensionality Reduction

- **PCA**: 95% variance retention
- Reduces features while preserving signal

## Hugging Face Integration

### Upload Model

```bash
python huggingface_integration/upload_model.py \
  --repo your-username/telecom-churn-model \
  --version 1.0.0
```

### Download Model

```bash
python huggingface_integration/download_model.py \
  --repo your-username/telecom-churn-model
```

### Interactive Demo

Try the model on [Hugging Face Spaces](https://huggingface.co/spaces/your-username/telecom-churn-demo)

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_ACCESS_KEY_ID` | AWS access key | - |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | - |
| `S3_BUCKET_NAME` | S3 bucket for DVC | `telecom-churn-bucket` |
| `MLFLOW_TRACKING_URI` | MLflow server URL | `http://localhost:5000` |
| `HF_TOKEN` | Hugging Face token | - |
| `API_HOST` | API bind address | `0.0.0.0` |
| `API_PORT` | API port | `8000` |

### Model Configuration

Edit `src/config.py` to adjust:
- `HIGH_VALUE_THRESHOLD`: Percentile for high-value (default: 0.70)
- `USE_SMOTE`: Enable/disable SMOTE
- `APPLY_PCA`: Enable/disable PCA
- Model hyperparameters

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test file
pytest tests/test_models.py -v
```

## Performance Metrics

Typical performance on test set:

| Metric | Score |
|--------|-------|
| AUC-ROC | 0.85+ |
| F1 Score | 0.75+ |
| Precision | 0.70+ |
| Recall | 0.80+ |

*Actual performance depends on data quality and feature engineering.*

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Data      │────▶│   DVC       │────▶│   AWS S3    │
│   Sources   │     │   Version   │     │   Storage   │
└─────────────┘     └─────────────┘     └─────────────┘
                              │
                              ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   FastAPI   │◀────│   Models    │◀────│   Training  │
│   Service   │     │   Registry  │     │   Pipeline  │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Docker    │     │   MLflow    │     │   GitHub    │
│   Container │     │   Tracking  │     │   Actions   │
└─────────────┘     └─────────────┘     └─────────────┘
                              │
                              ▼
                       ┌─────────────┐
                       │   Hugging   │
                       │   Face Hub  │
                       └─────────────┘
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit pull request

## License

MIT License - see [LICENSE](LICENSE) file.

## Citation

```bibtex
@software{telecom_churn_mlops,
  title = {MLOps-Driven End-to-End Pipeline for Telecom Churn Prediction},
  author = {MLOps Team},
  year = {2024},
  url = {https://github.com/your-org/telecom-churn-mlops}
}
```

## Contact

- GitHub Issues: [Report bugs or request features](https://github.com/your-org/telecom-churn-mlops/issues)
- Email: mlops-team@example.com

---

Built with modern MLOps practices for production ML systems.
