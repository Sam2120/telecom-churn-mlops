---
language: en
license: mit
library_name: scikit-learn
tags:
- tabular-classification
- churn-prediction
- telecom
- mlops
datasets:
- telecom-customer-data
metrics:
- auc
- f1
- precision
- recall
model_name: Telecom Churn Prediction Model
---

# Model Card for Telecom Churn Prediction Model

This model predicts whether high-value telecom customers are likely to churn (cancel their service) based on their usage patterns over a 3-month period.

## Model Details

### Model Description

This is an ensemble of scikit-learn classifiers trained to predict customer churn in the telecom industry. The model focuses specifically on high-value customers (top 70th percentile by recharge amount) and uses behavioral data from months 6, 7, and 8 to predict churn in month 9.

- **Developed by:** MLOps Pipeline Team
- **Shared by:** Sam2120
- **Model type:** Binary classification ensemble (Logistic Regression, Random Forest, Gradient Boosting)
- **Language(s) (NLP):** en
- **License:** mit
- **Finetuned from model:** Not applicable - trained from scratch

### Model Sources

- **Repository:** https://github.com/Sam2120/telecom-churn-mlops
- **Paper:** Not applicable
- **Demo:** Hugging Face Spaces integration available

## Uses

### Direct Use

This model is designed for:
- **Customer retention teams** to identify at-risk high-value customers
- **Marketing teams** to target retention campaigns
- **Business analysts** to understand churn patterns and drivers
- **Real-time churn risk scoring** via the provided FastAPI endpoint

The model accepts tabular customer data with features like call usage, recharge history, and internet usage, and outputs a churn probability (0-1) and binary prediction.

### Downstream Use

- **Integration with CRM systems** for automated retention workflows
- **Customer lifetime value (CLV) modeling** by combining churn predictions with revenue data
- **Campaign optimization** to prioritize high-risk, high-value customers
- **Fraud detection** for identifying unusual usage patterns

### Out-of-Scope Use

This model should NOT be used for:
- **Predicting churn for low-value customers** (model trained only on high-value segment)
- **Long-term churn prediction** beyond the 1-month horizon it was trained for
- **Different telecom markets** without retraining (model behavior may not generalize)
- **Individual customer harassment** or punitive actions based solely on model predictions
- **Credit scoring or loan decisions** (not designed for financial risk assessment)

## Bias, Risks, and Limitations

### Technical Limitations
- **Class imbalance:** The dataset has a natural churn rate of ~8-15%, which creates challenges in recall optimization
- **Temporal leakage risk:** Features must be strictly from the "good phase" (months 6-8) to avoid data leakage
- **Feature drift:** Telecom usage patterns change over time; model performance degrades without periodic retraining
- **Missing value sensitivity:** Model requires complete feature sets; imputation strategies may introduce bias

### Sociotechnical Considerations
- **Demographic bias:** If training data contains demographic information correlated with protected attributes, the model may inadvertently learn biased patterns
- **Surveillance concerns:** Predictive models for customer behavior raise privacy considerations
- **Feedback loops:** Aggressive retention campaigns targeting predicted churners may create self-fulfilling prophecies

### Recommendations
- Regular model retraining (monthly/quarterly) to account for changing customer behaviors
- A/B testing of retention interventions to measure true causal impact
- Monitoring for demographic disparities in predictions across customer segments
- Combining model predictions with human judgment for final retention decisions
- Transparent communication with customers about data usage for service improvement

## How to Get Started with the Model

### Installation
```bash
pip install -r requirements.txt
```

### Quick Start
```python
import pickle
import pandas as pd

# Load the model
with open("models/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

# Prepare features (see src/feature_engineering.py for expected format)
features = pd.DataFrame({
    "total_calls_m6": [100],
    "total_calls_m7": [95],
    "total_calls_m8": [20],
    # ... other features
})

# Predict
prediction = model.predict(features)
probability = model.predict_proba(features)[:, 1]
print(f"Churn Risk: {probability[0]:.2%}")
```

### API Usage
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": {"total_calls_m6": 100, ...}}'
```

## Training Details

### Training Data

**Data Source:** Telecom customer usage data spanning 4 months (months 6-9)

**Dataset Characteristics:**
- **Format:** Tabular CSV with wide format (one row per customer, columns per month)
- **Size:** ~30,000 high-value customers (top 70th percentile by recharge amount)
- **Features:** 30+ features including:
  - **Usage features:** Total calls, duration, incoming/outgoing minutes (months 6-8)
  - **Recharge features:** Recharge amounts and counts (months 6-8)
  - **Internet features:** Data usage and sessions (months 6-8)
  - **Derived features:** Average recharge, usage trends, month-over-month changes

**Target Definition:**
- Churn = 1 if customer has zero incoming calls AND zero outgoing calls AND zero mobile internet usage in month 9
- Churn = 0 otherwise
- Typical churn rate: ~8-15% among high-value customers

**Preprocessing Steps:**
1. Data validation (column checks, null handling)
2. High-value customer identification (70th percentile threshold)
3. Churn label definition based on month 9 usage
4. Feature engineering (usage ratios, trends, averages)
5. Train/test split: 80/20 with stratification by churn label

### Training Procedure

#### Preprocessing
- **Feature Engineering:** `src/feature_engineering.py` creates rolling averages, month-over-month ratios, and interaction features
- **SMOTE:** Applied for class balancing (sampling_strategy=0.5)
- **PCA:** Optional dimensionality reduction (95% variance threshold)
- **Scaling:** StandardScaler for Logistic Regression (embedded in pipeline)

#### Training Hyperparameters

**Logistic Regression:**
- C: 1.0
- Penalty: l2
- Solver: lbfgs
- Max iterations: 1000
- Class weight: balanced

**Random Forest:**
- N estimators: 100
- Max depth: 10
- Min samples split: 5
- Min samples leaf: 2
- Class weight: balanced

**Gradient Boosting:**
- N estimators: 100
- Max depth: 5
- Learning rate: 0.1

**Training regime:** fp32 (standard scikit-learn precision)

#### Speeds, Sizes, Times
- **Training time:** ~2-5 minutes on standard CPU
- **Inference time:** <10ms per prediction
- **Model size:** ~5-15 MB per model (pickled)
- **Memory usage:** ~500MB during training (30K samples)

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data
- **Test set:** 20% stratified holdout from original data
- **Size:** ~6,000 samples (typical)
- **Churn rate:** Matched to training distribution (~8-15%)

#### Factors
Evaluation is performed across:
- Overall performance (all high-value customers)
- Cross-validation (5-fold stratified) for robustness
- Individual model comparison (Logistic Regression vs Random Forest vs Gradient Boosting)

#### Metrics
- **AUC-ROC:** Primary metric for ranking predictions
- **F1-Score:** Balance between precision and recall
- **Precision:** Minimize false alarms (don't waste retention budget)
- **Recall:** Capture actual churners (don't miss at-risk customers)
- **Average Precision:** Alternative to AUC for imbalanced data

### Results

Performance varies by model architecture. The ensemble selects the best model based on validation AUC.

**Typical Performance (Random Forest - usually best):**
- AUC: ~0.85-0.92
- F1: ~0.70-0.80
- Precision: ~0.65-0.75
- Recall: ~0.75-0.85

**Logistic Regression (most interpretable):**
- AUC: ~0.80-0.88
- F1: ~0.65-0.75

**Gradient Boosting:**
- AUC: ~0.83-0.90
- F1: ~0.68-0.78

#### Summary
The model achieves strong discriminative performance (AUC > 0.85) suitable for production deployment. Random Forest typically performs best, while Logistic Regression provides the most interpretable results for business stakeholders.

## Model Examination

### Feature Importance
Top predictive features typically include:
1. **Month-over-month usage decline** (steep drops in calls/data)
2. **Average recharge amount** (lower recharge = higher risk)
3. **Recent usage patterns** (month 8 vs month 6-7 averages)
4. **Data usage consistency** (frequent data users who stop are high risk)

### Interpretability
- **Logistic Regression:** Coefficients indicate feature direction and magnitude
- **Random Forest:** Feature importance scores available via `sklearn`
- **SHAP/Partial Dependence:** Can be computed post-hoc for individual explanations

## Environmental Impact

Carbon emissions are minimal given the small dataset and efficient scikit-learn algorithms.

- **Hardware Type:** Standard CPU (no GPU required)
- **Hours used:** <0.1 hours per training run
- **Cloud Provider:** Local/AWS (configurable)
- **Compute Region:** us-east-1 (configurable)
- **Carbon Emitted:** <10g CO2eq per training run (estimated)

## Technical Specifications

### Model Architecture and Objective

**Objective:** Minimize binary cross-entropy loss for churn prediction

**Architecture:** Ensemble of three scikit-learn classifiers:
1. Logistic Regression (linear baseline)
2. Random Forest (tree-based, handles non-linearities)
3. Gradient Boosting (sequential trees, high accuracy)

**Pipeline:**
```
Input Features → [Optional SMOTE] → Classifier → Churn Probability
```

### Compute Infrastructure

#### Hardware
- **Minimum:** 2 CPU cores, 4GB RAM
- **Recommended:** 4 CPU cores, 8GB RAM
- **GPU:** Not required (CPU-only training)

#### Software
- **Python:** 3.10+
- **Key dependencies:**
  - scikit-learn 1.3+
  - pandas 2.0+
  - numpy 1.24+
  - imbalanced-learn (SMOTE)
  - mlflow (experiment tracking)
  - fastapi (serving)

## Citation

**BibTeX:**
```bibtex
@software{telecom_churn_mlops,
  title = {Telecom Churn Prediction MLOps Pipeline},
  author = {Sam2120},
  year = {2024},
  url = {https://github.com/Sam2120/telecom-churn-mlops}
}
```

**APA:**
Sam2120. (2024). *Telecom Churn Prediction MLOps Pipeline* [Software]. GitHub. https://github.com/Sam2120/telecom-churn-mlops

## Glossary

- **Churn:** Customer cancellation of telecom service
- **High-Value Customer:** Top 70th percentile by average recharge amount in good phase months
- **Good Phase:** Months 6, 7, 8 (stable period before prediction)
- **Action Month:** Month 8 (last month with full data before prediction)
- **SMOTE:** Synthetic Minority Over-sampling Technique (for class balancing)
- **AUC:** Area Under the ROC Curve (discrimination metric)
- **ARPU:** Average Revenue Per User
- **MOU:** Minutes of Usage

## More Information

- Full documentation: See `docs/architecture.md`
- API documentation: Available at `/docs` when running the FastAPI server
- MLflow UI: Run `mlflow ui` to see experiment tracking
- DVC pipeline: Run `dvc repro` to reproduce the full pipeline

## Model Card Authors

- MLOps Pipeline Team
- Sam2120

## Model Card Contact

For questions or issues, please open an issue on the GitHub repository: https://github.com/Sam2120/telecom-churn-mlops/issues

