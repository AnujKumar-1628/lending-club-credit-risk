import joblib

from credit_risk.data.load_data import load_cleaned_data
from credit_risk.data.split_data import DataSplitter
from credit_risk.features.build_features import FeatureBuilder
from credit_risk.evaluation.model_comparison import compare_models
from credit_risk.utils.logging import get_logger
from credit_risk.utils.paths import project_root


logger = get_logger(__name__)

# Paths to trained models
LOGISTIC_MODEL_PATH = project_root / "models" / "logistic" / "model.pkl"
XGBOOST_MODEL_PATH = project_root / "models" / "xgboost" / "model.pkl"


def main():
    logger.info("Starting model comparison")

    # -------------------------------------------------
    # Load cleaned data
    # -------------------------------------------------
    df = load_cleaned_data()
    logger.info(f"Loaded cleaned data: {df.shape}")

    # -------------------------------------------------
    # Time-based split
    # IMPORTANT: we need TRAIN + VAL
    # -------------------------------------------------
    splitter = DataSplitter()
    train_df, val_df, _ = splitter.split(df)

    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Validation shape: {val_df.shape}")

    # -------------------------------------------------
    # Feature engineering (MATCH TRAINING EXACTLY)
    # -------------------------------------------------
    feature_builder = FeatureBuilder()

    # Fit ONLY on training data
    X_train, y_train = feature_builder.build_features(train_df, fit=True)

    # Transform validation using same feature space
    X_val, y_val = feature_builder.build_features(val_df, fit=False)

    logger.info(f"X_val shape: {X_val.shape}")

    # -------------------------------------------------
    # Load trained models
    # -------------------------------------------------
    if not LOGISTIC_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Logistic model not found. Run train_logistic.py first."
        )

    if not XGBOOST_MODEL_PATH.exists():
        raise FileNotFoundError("XGBoost model not found. Run train_xgboost.py first.")

    logistic_model = joblib.load(LOGISTIC_MODEL_PATH)
    xgboost_model = joblib.load(XGBOOST_MODEL_PATH)

    logger.info("Models loaded successfully")

    # -------------------------------------------------
    # Compare models
    # -------------------------------------------------
    models = {
        "Logistic_SGD": logistic_model,
        "XGBoost": xgboost_model,
    }

    comparison_df = compare_models(
        models=models,
        X=X_val,
        y=y_val,
        threshold=0.5,
    )

    print("\nMODEL COMPARISON (VALIDATION SET)")
    print(comparison_df)

    logger.info("Model comparison completed successfully")


if __name__ == "__main__":
    main()
