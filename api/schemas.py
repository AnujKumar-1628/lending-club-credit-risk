"""
API Request/Response Schemas

Pydantic models for input validation and response serialization.
All fields are validated according to business rules and data constraints.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime
from enum import Enum
import re


# ============================
# Enums
# ============================
class RiskCategory(str, Enum):
    """Credit risk classification categories"""

    LOW = "Low Risk"
    MEDIUM = "Medium Risk"
    HIGH = "High Risk"


# ============================
# Input Schemas
# ============================
class LoanRequest(BaseModel):
    """
    Single loan application request.

    All fields must match what FeatureBuilder expects.
    Date fields must be in format 'Mon-YYYY' (e.g., 'Jan-2018').
    """

    # --------------------------------------------------------
    # Numeric features - fed to FeatureBuilder.num_features
    # --------------------------------------------------------
    loan_amnt: float = Field(
        ..., gt=0, le=100000, description="Loan amount in USD", examples=[10000.0]
    )
    int_rate: float = Field(
        ...,
        ge=0,
        le=35.0,
        description="Interest rate as percentage (0-35)",
        examples=[12.5],
    )
    installment: float = Field(
        ...,
        gt=0,
        le=10000,
        description="Monthly installment amount in USD",
        examples=[250.0],
    )
    annual_inc: float = Field(
        ..., gt=0, le=10000000, description="Annual income in USD", examples=[60000.0]
    )
    dti: float = Field(
        ...,
        ge=0,
        le=100,
        description="Debt-to-income ratio as percentage",
        examples=[18.0],
    )
    revol_bal: float = Field(
        ..., ge=0, description="Revolving balance in USD", examples=[5000.0]
    )
    revol_util: float = Field(
        ...,
        ge=0,
        le=150,
        description="Revolving utilization rate as percentage",
        examples=[35.0],
    )
    open_acc: int = Field(
        ..., ge=0, le=100, description="Number of open credit accounts", examples=[8]
    )
    total_acc: int = Field(
        ..., ge=0, le=200, description="Total number of credit accounts", examples=[20]
    )
    mort_acc: int = Field(
        ..., ge=0, le=50, description="Number of mortgage accounts", examples=[1]
    )
    emp_length_num: float = Field(
        ..., ge=0, le=50, description="Employment length in years", examples=[5.0]
    )

    # --------------------------------------------------------
    # Date strings - parsed by _add_core_features
    # Format: 'Mon-YYYY' (e.g., 'Jan-2018')
    # --------------------------------------------------------
    issue_d: str = Field(
        ...,
        min_length=8,
        max_length=8,
        description="Loan issue date in format 'Mon-YYYY'",
        examples=["Jan-2018"],
    )
    earliest_cr_line: str = Field(
        ...,
        min_length=8,
        max_length=8,
        description="Earliest credit line date in format 'Mon-YYYY'",
        examples=["May-2002"],
    )

    # --------------------------------------------------------
    # FICO scores - combined into fico_avg by pipeline
    # --------------------------------------------------------
    fico_range_low: int = Field(
        ...,
        ge=300,
        le=850,
        description="Lower bound of FICO score range",
        examples=[650],
    )
    fico_range_high: int = Field(
        ...,
        ge=300,
        le=850,
        description="Upper bound of FICO score range",
        examples=[700],
    )

    # --------------------------------------------------------
    # Categorical features - fed to FeatureBuilder.cat_features
    # --------------------------------------------------------
    term: str = Field(
        ...,
        pattern=r"^(36|60) months$",
        description="Loan term: '36 months' or '60 months'",
        examples=["36 months"],
    )
    addr_state: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="US state code (2 letters)",
        examples=["CA"],
    )
    home_ownership: str = Field(
        ..., description="Home ownership status", examples=["RENT"]
    )
    purpose: str = Field(
        ..., description="Loan purpose", examples=["debt_consolidation"]
    )
    verification_status: str = Field(
        ..., description="Income verification status", examples=["Verified"]
    )
    application_type: str = Field(
        ..., description="Individual or Joint application", examples=["Individual"]
    )
    initial_list_status: str = Field(
        ..., description="Initial listing status", examples=["w"]
    )
    sub_grade: str = Field(
        ...,
        pattern=r"^[A-G][1-5]$",
        description="Loan sub-grade (A1-G5)",
        examples=["B2"],
    )

    # --------------------------------------------------------
    # Binary flags - fed to FeatureBuilder.binary_features
    # --------------------------------------------------------
    pub_rec: int = Field(
        ..., ge=0, description="Number of derogatory public records", examples=[0]
    )
    pub_rec_bankruptcies: int = Field(
        ..., ge=0, description="Number of public record bankruptcies", examples=[0]
    )
    emp_length_missing: int = Field(
        ...,
        ge=0,
        le=1,
        description="Flag: 1 if employment length is missing, 0 otherwise",
        examples=[0],
    )
    revol_util_missing: int = Field(
        ...,
        ge=0,
        le=1,
        description="Flag: 1 if revolving utilization is missing, 0 otherwise",
        examples=[0],
    )
    mort_acc_missing: int = Field(
        ...,
        ge=0,
        le=1,
        description="Flag: 1 if mortgage account count is missing, 0 otherwise",
        examples=[0],
    )

    @field_validator("issue_d", "earliest_cr_line")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate date strings are in 'Mon-YYYY' format"""
        if not re.match(r"^[A-Z][a-z]{2}-\d{4}$", v):
            raise ValueError(
                f"Date must be in format 'Mon-YYYY' (e.g., 'Jan-2018'), got '{v}'"
            )
        return v

    @field_validator("fico_range_high")
    @classmethod
    def validate_fico_range(cls, v: int, info) -> int:
        """Ensure fico_range_high >= fico_range_low"""
        if "fico_range_low" in info.data and v < info.data["fico_range_low"]:
            raise ValueError(
                f"fico_range_high ({v}) must be >= fico_range_low ({info.data['fico_range_low']})"
            )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "loan_amnt": 10000.0,
                "int_rate": 12.5,
                "installment": 250.0,
                "annual_inc": 60000.0,
                "dti": 18.0,
                "revol_bal": 5000.0,
                "revol_util": 35.0,
                "open_acc": 8,
                "total_acc": 20,
                "mort_acc": 1,
                "emp_length_num": 5.0,
                "issue_d": "Jan-2018",
                "earliest_cr_line": "May-2002",
                "fico_range_low": 650,
                "fico_range_high": 700,
                "term": "36 months",
                "addr_state": "CA",
                "home_ownership": "RENT",
                "purpose": "debt_consolidation",
                "verification_status": "Verified",
                "application_type": "Individual",
                "initial_list_status": "w",
                "sub_grade": "B2",
                "pub_rec": 0,
                "pub_rec_bankruptcies": 0,
                "emp_length_missing": 0,
                "revol_util_missing": 0,
                "mort_acc_missing": 0,
            }
        }
    }


class BatchLoanRequest(BaseModel):
    """Batch prediction request containing multiple loans"""

    loans: List[LoanRequest] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of loan applications (max 1000 per batch)",
    )


# ============================
# Output Schemas
# ============================
class PredictionResponse(BaseModel):
    """Single loan prediction response"""

    loan_id: Optional[int] = Field(
        None, description="Loan identifier (null for single predictions)"
    )
    default_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Predicted probability of default [0.0, 1.0]"
    )
    default_prediction: int = Field(
        ..., ge=0, le=1, description="Binary prediction: 0 = no default, 1 = default"
    )
    risk_category: RiskCategory = Field(..., description="Risk bucket: Low/Medium/High")
    recommendation: str = Field(
        ..., description="Business recommendation: Approve/Review/Reject"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "loan_id": None,
                "default_probability": 0.2345,
                "default_prediction": 0,
                "risk_category": "Low Risk",
                "recommendation": "Approve",
            }
        }
    }


class BatchSummary(BaseModel):
    """Summary statistics for batch predictions"""

    total: int = Field(..., description="Total number of loans processed")
    approved: int = Field(..., description="Number recommended for approval")
    reviewed: int = Field(..., description="Number requiring manual review")
    rejected: int = Field(..., description="Number recommended for rejection")
    avg_default_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Average default probability across batch"
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction response with individual results and summary"""

    total_loans: int = Field(..., description="Total loans in batch")
    predictions: List[PredictionResponse] = Field(
        ..., description="Individual predictions for each loan"
    )
    summary: BatchSummary = Field(..., description="Aggregate statistics")


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Service status: 'healthy' or 'unhealthy'")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    model_name: str = Field(..., description="Active model name")
    api_version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Current server timestamp")


class ErrorResponse(BaseModel):
    """Standard error response"""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Additional error details")
