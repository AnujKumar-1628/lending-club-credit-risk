import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix


def ks_statistic(y_true, y_prob):
    data = pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).sort_values(
        "y_prob", ascending=False
    )

    data["cum_good"] = (data["y_true"] == 0).cumsum() / (data["y_true"] == 0).sum()
    data["cum_bad"] = (data["y_true"] == 1).cumsum() / (data["y_true"] == 1).sum()

    return (data["cum_bad"] - data["cum_good"]).abs().max()


def evaluate_classification(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "ks": ks_statistic(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
