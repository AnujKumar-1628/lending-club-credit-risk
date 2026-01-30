import pandas as pd

from credit_risk.evaluation.metrics import evaluate_classification
from credit_risk.utils.logging import get_logger

logger = get_logger(__name__)


def compare_models(models: dict, X, y, threshold: float = 0.5) -> pd.DataFrame:
    """
    Compare multiple trained models on the same dataset.

    Parameters
    ----------
    models : dict
        Dictionary of model_name -> trained model object
    X : array-like
        Feature matrix
    y : array-like
        True labels
    threshold : float
        Decision threshold for confusion matrix

    Returns
    -------
    pd.DataFrame
        Comparison table with ROC-AUC, KS, and confusion matrix
    """

    logger.info("Starting model comparison")

    results = []

    for name, model in models.items():
        logger.info(f"Evaluating model: {name}")

        y_prob = model.predict_proba(X)

        metrics = evaluate_classification(
            y_true=y,
            y_prob=y_prob,
            threshold=threshold,
        )

        results.append(
            {
                "model": name,
                "roc_auc": metrics["roc_auc"],
                "ks": metrics["ks"],
                "tn": metrics["confusion_matrix"][0, 0],
                "fp": metrics["confusion_matrix"][0, 1],
                "fn": metrics["confusion_matrix"][1, 0],
                "tp": metrics["confusion_matrix"][1, 1],
            }
        )

    comparison_df = pd.DataFrame(results).sort_values(by="ks", ascending=False)

    logger.info("Model comparison completed")
    return comparison_df
