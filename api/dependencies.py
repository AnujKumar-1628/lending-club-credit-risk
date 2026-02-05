import joblib
from pathlib import Path
from functools import lru_cache
from typing import Dict, Any

from credit_risk.utils.logging import get_logger

logger = get_logger(__name__)

# -------------------------------------------------
# Configuration (change here if needed)
# -------------------------------------------------
MODEL_NAME = "xgboost"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / MODEL_NAME

MODEL_PATH = MODEL_DIR / "model.pkl"
FEATURE_BUILDER_PATH = MODEL_DIR / "feature_builder.pkl"


# -------------------------------------------------
# Load trained artifacts ONCE
# -------------------------------------------------
@lru_cache(maxsize=1)
def load_artifacts() -> Dict[str, Any]:
    """
    Loads model and feature builder once.
    Reused for every API request.
    """

    logger.info("Loading model artifacts for API")

    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.pkl not found. Run training first.")

    if not FEATURE_BUILDER_PATH.exists():
        raise FileNotFoundError("feature_builder.pkl not found. Run training first.")

    model = joblib.load(MODEL_PATH)
    feature_builder = joblib.load(FEATURE_BUILDER_PATH)

    logger.info("Model and FeatureBuilder loaded successfully")

    return {
        "model": model,
        "feature_builder": feature_builder,
        "model_name": MODEL_NAME,
    }


# -------------------------------------------------
# FastAPI dependency
# -------------------------------------------------
def get_artifacts() -> Dict[str, Any]:
    return load_artifacts()
