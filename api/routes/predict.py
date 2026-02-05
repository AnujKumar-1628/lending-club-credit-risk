import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import APIRouter, Depends

from api.schemas import (
    LoanRequest,
    BatchLoanRequest,
    PredictionResponse,
    BatchPredictionResponse,
    BatchSummary,
    HealthResponse,
    RiskCategory,
)
from api.dependencies import get_artifacts
from credit_risk.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# -------------------------------------------------
# Business thresholds
# -------------------------------------------------
LOW_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.60


def classify_risk(prob: float) -> RiskCategory:
    if prob < LOW_RISK_THRESHOLD:
        return RiskCategory.LOW
    elif prob < HIGH_RISK_THRESHOLD:
        return RiskCategory.MEDIUM
    else:
        return RiskCategory.HIGH


def decision(prob: float) -> str:
    if prob < LOW_RISK_THRESHOLD:
        return "Approve"
    elif prob < HIGH_RISK_THRESHOLD:
        return "Review"
    else:
        return "Reject"


# -------------------------------------------------
# SINGLE LOAN PREDICTION
# -------------------------------------------------
@router.post("/predict", response_model=PredictionResponse)
def predict_single(
    loan: LoanRequest,
    artifacts: dict = Depends(get_artifacts),
):
    logger.info("Received single prediction request")

    # 1. Convert request to DataFrame
    df = pd.DataFrame([loan.model_dump()])

    # 2. Feature engineering
    feature_builder = artifacts["feature_builder"]
    X, _ = feature_builder.build_features(df, fit=False)

    # 3. Model prediction
    model = artifacts["model"]
    prob = float(model.predict_proba(X)[0, 1])

    # 4. Build response
    return PredictionResponse(
        loan_id=None,
        default_probability=round(prob, 4),
        default_prediction=int(prob >= 0.5),
        risk_category=classify_risk(prob),
        recommendation=decision(prob),
    )


# -------------------------------------------------
# BATCH LOAN PREDICTION
# -------------------------------------------------
@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    batch: BatchLoanRequest,
    artifacts: dict = Depends(get_artifacts),
):
    logger.info(f"Received batch request with {len(batch.loans)} loans")

    df = pd.DataFrame([loan.model_dump() for loan in batch.loans])

    feature_builder = artifacts["feature_builder"]
    X, _ = feature_builder.build_features(df, fit=False)

    model = artifacts["model"]
    probs = model.predict_proba(X)[:, 1]

    predictions = []
    for idx, prob in enumerate(probs):
        predictions.append(
            PredictionResponse(
                loan_id=idx,
                default_probability=round(float(prob), 4),
                default_prediction=int(prob >= 0.5),
                risk_category=classify_risk(prob),
                recommendation=decision(prob),
            )
        )

    summary = BatchSummary(
        total=len(predictions),
        approved=sum(p.recommendation == "Approve" for p in predictions),
        reviewed=sum(p.recommendation == "Review" for p in predictions),
        rejected=sum(p.recommendation == "Reject" for p in predictions),
        avg_default_probability=round(float(np.mean(probs)), 4),
    )

    return BatchPredictionResponse(
        total_loans=len(predictions),
        predictions=predictions,
        summary=summary,
    )


# -------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------
@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name="xgboost",
        api_version="1.0.0",
        timestamp=datetime.utcnow(),
    )
