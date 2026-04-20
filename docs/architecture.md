# Architecture Documentation

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA SOURCES                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Telecom   │  │   Recharge  │  │    Call     │  │    Data     │        │
│  │   Billing   │  │   Records   │  │   Records   │  │   Usage     │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
└─────────┼────────────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA PIPELINE                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         AWS S3 Storage                              │    │
│  │                     (Raw Data Landing Zone)                         │    │
│  └─────────────────────────────────┬───────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         DVC Data Versioning                           │    │
│  │              • Git-tracked data versioning                          │    │
│  │              • Remote storage on S3                                 │    │
│  │              • Reproducible data lineage                            │    │
│  └─────────────────────────────────┬───────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    DVC Pipeline Stages                              │    │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌────────┐ │    │
│  │  │  Load   │──▶│Preproc. │──▶│  Split  │──▶│ Train   │──▶│Evaluate│ │    │
│  │  │  Data   │   │Features │   │  Data   │   │ Models  │   │        │ │    │
│  │  └─────────┘   └─────────┘   └─────────┘   └─────────┘   └────────┘ │    │
│  └─────────────────────────────────┬───────────────────────────────────┘    │
└────────────────────────────────────┼────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ML PLATFORM                                      │
│                                                                             │
│  ┌─────────────────────────────┐    ┌─────────────────────────────────────┐  │
│  │      MLflow Tracking        │    │         Model Registry              │  │
│  │  ┌─────────────────────┐    │    │  ┌─────────────────────────────┐  │  │
│  │  │  Experiments        │    │    │  │  • Version Management         │  │  │
│  │  │  • Parameters       │    │    │  │  • Stage Transitions          │  │  │
│  │  │  • Metrics (AUC)    │    │    │  │  • Production Promotions      │  │  │
│  │  │  • Artifacts        │    │    │  │  • Rollback Capabilities      │  │  │
│  │  └─────────────────────┘    │    │  └─────────────────────────────┘  │  │
│  └─────────────────────────────┘    └─────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Model Training                                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │   Logistic   │  │    Random    │  │   Gradient   │               │   │
│  │  │  Regression  │  │    Forest    │  │   Boosting   │               │   │
│  │  │              │  │              │  │              │               │   │
│  │  │ Interpretable│  │  Best AUC    │  │  Accuracy    │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  │                                                                     │   │
│  │  Feature Engineering:                                               │   │
│  │  • SMOTE (Class Imbalance)                                         │   │
│  │  • PCA (Dimensionality Reduction)                                  │   │
│  └─────────────────────────────────┬───────────────────────────────────┘   │
└────────────────────────────────────┼──────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DEPLOYMENT LAYER                                    │
│                                                                             │
│  ┌───────────────────────────┐    ┌───────────────────────────────────────┐   │
│  │    Docker Container       │    │        FastAPI Service              │   │
│  │                           │    │                                       │   │
│  │  ┌─────────────────────┐  │    │  ┌─────────┐ ┌─────────┐ ┌────────┐ │   │
│  │  │  Scikit-learn Model │  │    │  │ /health │ │/predict │ │/explain│ │   │
│  │  │  PCA Transformer    │  │    │  └─────────┘ └─────────┘ └────────┘ │   │
│  │  │  Preprocessor       │  │    │                                       │   │
│  │  └─────────────────────┘  │    │  Features:                            │   │
│  │                           │    │  • Single Prediction                  │   │
│  │  Port: 8000               │    │  • Batch Prediction                   │   │
│  │  Health Check Enabled     │    │  • File Upload                        │   │
│  └───────────────────────────┘    │  • Explanations                       │   │
│                                     └───────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CI/CD PIPELINE                                      │
│                                                                             │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
│  │   Test  │──▶│  Data   │──▶│  Train  │──▶│  Build  │──▶│ Deploy  │     │
│  │  Code   │   │  DVC    │   │ MLflow  │   │ Docker  │   │  ECS    │     │
│  │ Quality │   │ Pipeline│   │ Tracking│   │  Image  │   │         │     │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘     │
│       │                                                        │            │
│       ▼                                                        ▼            │
│  ┌─────────┐                                              ┌─────────┐      │
│  │  Unit   │                                              │ Hugging │      │
│  │  Tests  │                                              │  Face   │      │
│  │ Coverage│                                              │  Hub    │      │
│  └─────────┘                                              └─────────┘      │
│                                                                             │
│  Platform: GitHub Actions                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Layer

**AWS S3**: Stores raw and processed data with versioning via DVC
- Raw data: `s3://telecom-churn-bucket/data/raw/`
- Processed data: `s3://telecom-churn-bucket/data/processed/`
- Model artifacts: `s3://telecom-churn-bucket/models/`

**DVC**: Data version control integrated with Git
- Tracks data changes alongside code
- Enables reproducible experiments
- Remote storage on S3

### 2. ML Platform

**MLflow**: Complete ML lifecycle management
- Experiment tracking with parameters and metrics
- Model registry with version staging
- Artifact storage for models and plots

**Training Pipeline**:
- SMOTE for handling class imbalance
- PCA for dimensionality reduction
- Multiple models: Logistic Regression, Random Forest, Gradient Boosting

### 3. Serving Layer

**FastAPI**: High-performance REST API
- Single prediction endpoint
- Batch prediction support
- File upload for CSV processing
- Prediction explanations

**Docker**: Containerization
- Multi-stage build for optimization
- Health checks for monitoring
- Non-root user for security

### 4. CI/CD Pipeline

**GitHub Actions**: Automated workflows
- Code quality checks (linting, formatting)
- Unit test execution with coverage
- DVC pipeline reproduction
- MLflow experiment tracking
- Docker image build and push
- Staging and production deployment
- Hugging Face model upload

## Data Flow

```
1. Raw Data → S3 Landing Zone
          ↓
2. DVC Pull → Local Workspace
          ↓
3. Load & Validate → DataLoader
          ↓
4. Feature Engineering → Transformers
          ↓
5. Train/Test Split → Stratified
          ↓
6. Model Training → MLflow Tracking
          ↓
7. Model Evaluation → Metrics
          ↓
8. Registry → Staging → Production
          ↓
9. Deployment → FastAPI Container
          ↓
10. Monitoring → Logs & Metrics
```

## Security Considerations

- Environment variables for secrets (AWS, Hugging Face)
- No credentials in code or Docker images
- S3 bucket policies for access control
- Docker non-root user execution
- API input validation with Pydantic

## Scalability

- Horizontal scaling with Docker containers
- S3 for unlimited storage
- MLflow server can be externalized
- Batch prediction support for large datasets
- PCA reduces feature dimensionality for faster inference
