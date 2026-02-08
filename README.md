# Lending Club Credit Risk Prediction

An end-to-end machine learning system that predicts loan default risk using historical LendingClub data. The project covers data cleaning, time-based splitting, reusable feature engineering, model training, evaluation, artifact versioning, and a FastAPI inference service with Docker deployment.

## Why this project
- End-to-end ML pipeline with reproducible artifacts and model cards
- Leakage-aware time-based train/validation split
- Consistent feature engineering between training and inference
- API-ready inference with input validation and risk recommendations
- Dockerized deployment

## Tech stack
- Python, pandas, scikit-learn, XGBoost
- FastAPI, Pydantic, Uvicorn
- Docker

## Problem statement
Lenders need to estimate the probability that a loan will default at application time.

Target definition:
- `1` = Charged Off (default)
- `0` = Fully Paid

Only loans with final outcomes are used for training. Ongoing loans (for example, `Current`) are excluded.

## Dataset
LendingClub accepted loans (2007 to 2018Q4).

Expected raw file:
- `data/raw/accepted_2007_to_2018Q4.csv`

## Modeling approach
Two models are trained and compared:
- Logistic Regression (SGD) as a baseline
- XGBoost as the performance-oriented model

Validation metrics from `models/*/metrics.json`:

| Model | ROC AUC | KS |
| --- | ---: | ---: |
| Logistic (SGD) | 0.6945 | 0.2794 |
| XGBoost | 0.7038 | 0.2928 |

## Project structure
- `api/` FastAPI app, routes, schemas, dependencies
- `src/credit_risk/` core package (`data`, `features`, `models`, `evaluation`, `utils`)
- `scripts/` pipeline scripts (`run_split.py`, `train_logistic.py`, `train_xgboost.py`, `compare_models.py`)
- `models/` saved model artifacts and model cards
- `tests/` unit and API tests
- `docs/` project documentation

Structured tree:

```text
lendingclub-credit-risk/
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
## Project architecture

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

## Quickstart
1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add the dataset file to `data/raw/accepted_2007_to_2018Q4.csv`.
4. Run data pipeline:

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
Run the API:

```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:
- `GET /health`
- `POST /predict`
- `POST /predict/batch`

Risk categories:
- Low Risk (`< 0.30`) -> `Approve`
- Medium Risk (`0.30` to `< 0.60`) -> `Review`
- High Risk (`>= 0.60`) -> `Reject`

## Docker
Build and run:

```bash
docker build -t anujmahlawat/credit-risk-api .
docker run -p 8000:8000 anujmahlawat/credit-risk-api
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

This allows inference and review without retraining.

## Limitations and future work
- No experiment tracking or model registry
- No drift monitoring
- No cost-sensitive threshold optimization
- No bias or fairness analysis

## License
MIT

