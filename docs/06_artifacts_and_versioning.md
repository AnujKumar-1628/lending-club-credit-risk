# Model Artifacts and Versioning

## Stored Artifacts
For each model, the following artifacts are saved:
- trained model file
- feature builder
- evaluation metrics
- model card
- auxiliary metadata (e.g. feature importance for XGBoost)

## Artifact Storage Decision
For this portfolio project, model artifacts are committed to the repository to allow reviewers to:
- run inference without retraining
- inspect outputs easily

In real production systems, model artifacts would be stored in a model registry or object storage instead of Git.

## Compatibility
Model metadata documents the training environment and compatible inference ranges to reduce version-related issues.

