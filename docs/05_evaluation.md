# Model Evaluation

Models are evaluated on a validation dataset using:

- ROC AUC
- Precision
- Recall
- F1-score
- ksi 

ROC AUC is mainly used for model comparison.

## Notes
- Probability threshold selection is not optimized
- Business cost trade-offs (false positives vs false negatives) are not modeled
- Metrics are stored as JSON files for reproducibility

