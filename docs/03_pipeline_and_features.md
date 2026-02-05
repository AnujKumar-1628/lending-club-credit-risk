# Data Pipeline and Feature Engineering

## Data Pipeline
- Data is loaded and cleaned using explicit preprocessing steps
- Missing and invalid values are handled before modeling

## Train–Validation Split
- A time-based split is used instead of random splitting
- Earlier loans are used for training, later loans for validation
- This helps reduce data leakage and better reflects real-world usage

## Feature Engineering
- Feature transformations are handled by a reusable feature builder
- The feature builder is:
  - fitted only on training data
  - saved as an artifact
  - reused during inference

This ensures consistency between training and serving.

