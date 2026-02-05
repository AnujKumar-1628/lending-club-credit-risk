# Model Card – Logistic Regression (Credit Risk)

## Model Overview

This model is a **logistic regression classifier** trained to predict the probability of loan default using historical loan data.

It is used as a **baseline model** in the credit risk modeling pipeline.

---

## Problem Type

- **Task:** Binary classification  
- **Target:** Loan default (default vs non-default)  
- **Output:** Probability of default  

---

## Training Data

- Source: LendingClub historical loan dataset  
- Data type: Structured tabular data  
- Time-based split was used for training and validation  

Only features that would be available at loan approval time were used to avoid data leakage.

---

## Feature Engineering

- Feature transformations are handled by a reusable feature builder  
- The feature builder was:
  - fitted on training data
  - saved as an artifact
  - reused during inference  

This ensures consistency between training and inference.

---

## Model Details

- Algorithm: Logistic Regression with SGD 
- Library: scikit-learn  
- Regularization: Default scikit-learn settings  
- Output: Probability score between 0 and 1  

Logistic regression was chosen for its simplicity and interpretability.

---

## Evaluation Metrics

The model was evaluated on a validation dataset using:
- ROC AUC  
- Precision  
- Recall  
- F1-score  

Exact metric values are stored in `metrics.json`.

---

## Intended Use

- Baseline model for credit risk prediction  
- Educational and portfolio purposes  


---

## Limitations

- Linear model, limited ability to capture complex interactions  
- No optimized decision threshold  
- Not calibrated for business-specific cost trade-offs  

---

## Ethical and Usage Considerations

- Predictions should not be used as the sole decision factor  
- This model is not intended for real lending decisions  
- Bias and fairness analysis has not been performed  

---

## Deployment Notes

- Model is serialized using `joblib`  
- Designed to be served via a FastAPI inference service  
- Dependencies must remain within compatible version ranges  

---

## Disclaimer

This model is part of a learning and portfolio project and is not production-ready.

