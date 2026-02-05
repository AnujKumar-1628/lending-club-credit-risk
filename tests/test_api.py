import numpy as np
from fastapi.testclient import TestClient

from api.app import create_app
from api.dependencies import get_artifacts


class DummyFeatureBuilder:
    def build_features(self, df, fit=False):
        return np.zeros((len(df), 1)), None


class DummyModel:
    def predict_proba(self, X):
        prob = 0.2
        return np.column_stack([1 - prob, np.full(len(X), prob)])


def _sample_payload():
    return {
        "issue_d": "Jan-2018",
        "earliest_cr_line": "May-2002",
        "fico_range_low": 650,
        "fico_range_high": 700,
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
        "pub_rec": 0,
        "pub_rec_bankruptcies": 0,
        "emp_length_missing": 0,
        "revol_util_missing": 0,
        "mort_acc_missing": 0,
        "term": "36 months",
        "grade": "B",
        "sub_grade": "B2",
        "home_ownership": "RENT",
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "application_type": "Individual",
        "initial_list_status": "w",
        "addr_state": "CA",
    }


def test_health_endpoint():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_name"] == "xgboost"


def test_predict_endpoint_with_mocked_artifacts():
    app = create_app()
    app.dependency_overrides[get_artifacts] = lambda: {
        "model": DummyModel(),
        "feature_builder": DummyFeatureBuilder(),
        "model_name": "xgboost",
    }

    client = TestClient(app)
    resp = client.post("/predict", json=_sample_payload())

    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["default_probability"] <= 1.0
    assert data["risk_category"] in {"Low Risk", "Medium Risk", "High Risk"}
