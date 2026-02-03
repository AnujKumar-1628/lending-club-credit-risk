"""
Model comparison utilities


"""

import pandas as pd
from typing import Dict

from credit_risk.evaluation.metrics import evaluate_classification
from credit_risk.utils.logging import get_logger

logger = get_logger(__name__)


def compare_models(
    models: Dict[str, object],
    X,
    y,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compare multiple classification models on the same dataset.

    Parameters
    ----------
    models : dict
        Dictionary of model_name -> trained model object
    X : array-like
        Feature matrix
    y : array-like
        True labels
    threshold : float
        Classification threshold

    Returns
    -------
    pd.DataFrame
        Comparison table with metrics
    """

    results = []

    for model_name, model in models.items():
        logger.info(f"Evaluating model: {model_name}")

        # Predict probabilities (STANDARDIZED CONTRACT)

        y_prob_2d = model.predict_proba(X)

        # Slice positive class probability
        y_prob = y_prob_2d[:, 1]

        # Evaluate metrics

        metrics = evaluate_classification(
            y_true=y,
            y_prob=y_prob,
            threshold=threshold,
        )

        results.append(
            {
                "model": model_name,
                "roc_auc": metrics["roc_auc"],
                "ks": metrics["ks"],
            }
        )

    comparison_df = pd.DataFrame(results).sort_values(by="roc_auc", ascending=False)

    return comparison_df.reset_index(drop=True)
