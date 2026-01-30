import json
import joblib

from credit_risk.data.load_data import load_cleaned_data
from credit_risk.data.split_data import DataSplitter
from credit_risk.features.build_features import FeatureBuilder
from credit_risk.models.xgboost_model import XGBoostModel
from credit_risk.models.train import train_model
from credit_risk.evaluation.metrics import evaluate_classification
from credit_risk.utils.logging import get_logger
from credit_risk.utils.paths import project_root

logger = get_logger(__name__)

MODEL_DIR = project_root / "models" / "xgboost"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"


def main():
    logger.info("Loading cleaned dataset")
    df = load_cleaned_data()

    splitter = DataSplitter()
    train_df, val_df, _ = splitter.split(df)

    feature_builder = FeatureBuilder()
    X_train, y_train = feature_builder.build_features(train_df, fit=True)
    X_val, y_val = feature_builder.build_features(val_df, fit=False)

    model = XGBoostModel()
    model = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    y_val_proba = model.predict_proba(X_val)

    metrics = evaluate_classification(
        y_true=y_val,
        y_prob=y_val_proba,
        threshold=0.5,
    )

    with open(METRICS_PATH, "w") as f:
        json.dump(
            {
                "roc_auc": metrics["roc_auc"],
                "ks": metrics["ks"],
                "confusion_matrix": metrics["confusion_matrix"].tolist(),
            },
            f,
            indent=4,
        )

    joblib.dump(model, MODEL_PATH)

    logger.info("XGBoost training pipeline completed successfully")


if __name__ == "__main__":
    main()
