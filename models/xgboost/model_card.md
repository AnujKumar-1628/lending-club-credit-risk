# Model Card – XGBoost (Credit Risk)

## Model Overview

This model is an **XGBoost-based binary classifier** trained to predict the probability of loan default using historical loan data.

It is the **primary performance-oriented model** in this project.

---

## Problem Type

- **Task:** Binary classification  
- **Target:** Loan default (default vs non-default)  
- **Output:** Probability of default  

---

## Training Data

- Source: LendingClub historical loan dataset  
- Data type: Structured tabular data  
- Time-based split used for training and validation  

Only features available at the time of loan approval were included.

---

## Feature Engineering

- Feature engineering is handled by a saved feature builder  
- Same feature transformations are used during training and inference  
- Feature importance is exported as `feature_importance.csv`  

This helps interpret model behavior at a high level.

---

## Model Details

- Algorithm: XGBoost Gradient Boosted Trees  
- Library: xgboost  
- Objective: Binary classification  
- Output: Probability score between 0 and 1  

XGBoost was chosen for its strong performance on tabular data and ability to model non-linear relationships.

---

## Evaluation Metrics

The model was evaluated using:
- ROC AUC  
- Precision  
- Recall  
- F1-score  

Metric results are stored in `metrics.json`.

---

## Intended Use

- Primary predictive model for credit risk scoring  
- Demonstration of a non-linear ML model  
- Portfolio and learning purposes  

---

## Limitations

- Less interpretable than logistic regression  
- No hyperparameter tuning beyond basic configuration  
- No probability calibration applied  
- No business-specific threshold optimization  

---

## Ethical and Usage Considerations

- Model predictions should not be used for real financial decisions  
- Bias and fairness analysis has not been conducted  
- Output probabilities require human interpretation  

---

## Deployment Notes

- Model is serialized and stored as an artifact  
- Served via a FastAPI inference service  
- Dependency compatibility must be maintained  

---

## Disclaimer

This model is part of a learning and portfolio project and is not intended for production use.

