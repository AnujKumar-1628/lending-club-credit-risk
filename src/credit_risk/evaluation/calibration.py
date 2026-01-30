import numpy as np
from sklearn.calibration import calibration_curve


def calibration_data(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )

    return {
        "mean_predicted_prob": prob_pred,
        "fraction_positives": prob_true,
    }
