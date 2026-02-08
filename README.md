# Lending Club Credit Risk Prediction

An end-to-end machine learning system that predicts loan default risk using historical LendingClub data. This project goes beyond model training to show a production-style pipeline: data cleaning, time-based splits, reusable feature engineering, model training and evaluation, artifact versioning, and a FastAPI inference service with Docker deployment.

**Why this stands out**
- End-to-end ML pipeline with reproducible artifacts and model cards
- Leakage-aware, time-based train/validation split
- Consistent feature engineering between training and inference
- Production-style API with input validation and risk-based decisions
- Dockerized deployment for portable inference

**Tech stack**
- Python, pandas, scikit-learn, XGBoost
- FastAPI, Pydantic, Uvicorn
- Docker

**Problem statement**
Lenders need to estimate the probability that a loan will default at application time. This is framed as a binary classification problem:
- `1` = Charged Off (default)
- `0` = Fully Paid

Only loans with final outcomes are used for training. Ongoing loans (for example, "Current") are excluded to avoid label uncertainty.

**Project Structure**

```text
lending-club-credit-risk/
|
+-- api/                          # FastAPI application
|   +-- __init__.py
|   +-- app.py                    # Main FastAPI app
|   +-- dependencies.py           # Dependency injection (load model artifacts)
|   +-- schemas.py                # Pydantic models for request/response
|   +-- routes/
|       +-- __init__.py
|       +-- predict.py            # Prediction endpoints
|
+-- src/credit_risk/              # Core ML package
|   +-- data/                     # Data processing modules
|   |   +-- load_data.py          # Data loading utilities
|   |   +-- clean_data.py         # Data cleaning logic
|   |   +-- split_data.py         # Train-validation splitting
|   |
|   +-- features/                 # Feature engineering
|   |   +-- build_features.py     # FeatureBuilder class
|   |
|   +-- models/                   # Model definitions
|   |   +-- base.py               # Base model interface
|   |   +-- logistic_model.py     # Logistic regression wrapper
|   |   +-- xgboost_model.py      # XGBoost wrapper
|   |   +-- train.py              # Training utilities
|   |   +-- predict.py            # Prediction utilities
|   |
|   +-- evaluation/               # Model evaluation
|   |   +-- metrics.py            # Metric calculations
|   |   +-- calibration.py        # Probability calibration
|   |   +-- model_comparison.py   # Compare multiple models
|   |
|   +-- utils/                    # Shared utilities
|       +-- config.py             # Configuration management
|       +-- logging.py            # Logging setup
|       +-- paths.py              # Path constants
|
+-- scripts/                      # Training scripts
|   +-- train_logistic.py         # Train logistic regression
|   +-- train_xgboost.py          # Train XGBoost
|   +-- compare_models.py         # Compare model performance
|   +-- run_split.py              # Generate train/val split
|
+-- models/                       # Saved model artifacts
|   +-- logistic/
|   |   +-- model.pkl
|   |   +-- feature_builder.pkl
|   |   +-- metrics.json
|   |   +-- model_card.md
|   +-- xgboost/
|       +-- model.pkl
|       +-- feature_builder.pkl
|       +-- metrics.json
|       +-- feature_importance.csv
|       +-- model_card.md
|
+-- notebooks/                    # Jupyter notebooks
|   +-- exploration/              # EDA notebooks
|   +-- feature_analysis/         # Feature engineering experiments
|   +-- modeling/                 # Model training experiments
|   +-- validation/               # Model validation
|
+-- tests/                        # Test suite
|   +-- test_api.py               # API endpoint tests
|   +-- test_features.py          # Feature engineering tests
|   +-- test_logistic_model.py    # Logistic model tests
|   +-- test_xgboost_model.py     # XGBoost model tests
|   +-- test_split.py             # Data splitting tests
|
+-- docs/                         # Documentation
|   +-- 00_overview.md
|   +-- 01_business_problem.md
|   +-- 02_target_definition.md
|   +-- 03_pipeline_and_features.md
|   +-- 04_models.md
|   +-- 05_evaluation.md
|   +-- 06_artifacts_and_versioning.md
|   +-- 07_inference_api.md
|   +-- 08_docker_and_environment.md
|   +-- 09_testing.md
|   +-- 10_limitations_and_future_work.md
|
+-- .github/
|   +-- workflows/
|       +-- ci.yml                # GitHub Actions CI pipeline
|
+-- Dockerfile                    # Docker configuration
+-- requirements.txt              # Production dependencies
+-- requirements-dev.txt          # Development dependencies
+-- setup.py                      # Package installation
+-- README.md                     # This file
+-- LICENSE                       # MIT License
```

**Project Architecture**
```text
[Data Layer]
Raw CSV -> Cleaning -> Time-based Split -> Feature Builder
    |
    v
[Modeling Layer]
Logistic SGD (baseline) + XGBoost (primary)
    |
    v
[Evaluation Layer]
ROC-AUC, KS, confusion matrix, model comparison
    |
    v
[Serving Layer]
FastAPI endpoints: /predict, /predict/batch, /health
    |
    v
[Deployment Layer]
Docker image + CI workflow
```
**Dataset**
LendingClub accepted loans (2007 to 2018Q4). Raw file expected at:
- `data/raw/accepted_2007_to_2018Q4.csv` 

## Modeling approach
Two models are trained and compared:
- Logistic Regression (SGD) as a transparent baseline
- XGBoost as the performance-oriented model

**Validation metrics (stored in `models/*/metrics.json`)**
| Model | ROC AUC | KS |
| --- | --- | --- |
| Logistic (SGD) | 0.6945 | 0.2794 |
| XGBoost | 0.7038 | 0.2928 |

Metrics are computed on the validation split with a default threshold of 0.5.

## Project structure
- `api/` FastAPI app and prediction routes
- `data/` raw, processed, and sample splits
- `docs/` design notes and project documentation
- `models/` trained model artifacts and model cards
- `scripts/` pipeline scripts (split, train, compare)
- `src/credit_risk/` core pipeline, features, models, evaluation, utils
- `tests/` unit and API tests

## Quickstart
1. Create and activate a virtual environment
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Add the raw dataset:
```bash
mkdir -p data/raw
# Place accepted_2007_to_2018Q4.csv into data/raw/
```
4. Run the data pipeline (clean + split):
```bash
python scripts/run_split.py
```
5. Train models:
```bash
python scripts/train_logistic.py
python scripts/train_xgboost.py
```
6. Compare models:
```bash
python scripts/compare_models.py
```

## Inference API
Start the API:
```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:
- `GET /health`
- `POST /predict`
- `POST /predict/batch`

Risk Categories:

Low Risk (< 30% default probability) → Approve
Medium Risk (30-60% default probability) → Review
High Risk (> 60% default probability) → Reject
```

## Docker
Build and run the container:
```bash
docker build -t  anujmahlawat/credit-risk-api .
docker run -p 8000:8000  anujmahlawat/credit-risk-api
```


## Testing
```bash
pytest
```

## Artifacts and reproducibility
For each model, the repo stores:
- serialized model
- fitted feature builder
- evaluation metrics
- model card

This enables inference and review without retraining.

## Limitations and future work
- No experiment tracking or model registry
- No drift monitoring
- No cost-sensitive threshold optimization
- No bias or fairness analysis

## License
MIT
