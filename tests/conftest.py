import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_cleaned_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_rows = 600

    issue_dates = pd.date_range("2014-01-01", periods=n_rows, freq="MS")
    earliest_dates = pd.date_range("2000-01-01", periods=n_rows, freq="MS")

    df = pd.DataFrame(
        {
            "issue_d": issue_dates.strftime("%b-%Y"),
            "earliest_cr_line": earliest_dates.strftime("%b-%Y"),
            "fico_range_low": rng.integers(600, 760, size=n_rows),
            "fico_range_high": rng.integers(700, 851, size=n_rows),
            "loan_amnt": rng.uniform(1000, 35000, size=n_rows),
            "int_rate": rng.uniform(5, 25, size=n_rows),
            "installment": rng.uniform(50, 1200, size=n_rows),
            "annual_inc": rng.uniform(25000, 180000, size=n_rows),
            "dti": rng.uniform(0, 35, size=n_rows),
            "revol_bal": rng.uniform(0, 50000, size=n_rows),
            "revol_util": rng.uniform(0, 100, size=n_rows),
            "open_acc": rng.integers(1, 20, size=n_rows),
            "total_acc": rng.integers(5, 80, size=n_rows),
            "mort_acc": rng.integers(0, 8, size=n_rows),
            "emp_length_num": rng.uniform(0, 20, size=n_rows),
            "pub_rec": rng.integers(0, 2, size=n_rows),
            "pub_rec_bankruptcies": rng.integers(0, 2, size=n_rows),
            "emp_length_missing": rng.integers(0, 2, size=n_rows),
            "revol_util_missing": rng.integers(0, 2, size=n_rows),
            "mort_acc_missing": rng.integers(0, 2, size=n_rows),
            "term": rng.choice(["36 months", "60 months"], size=n_rows),
            "addr_state": rng.choice(["CA", "NY", "TX", "FL", "IL", "WA"], size=n_rows),
            "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN"], size=n_rows),
            "purpose": rng.choice(
                ["debt_consolidation", "credit_card", "home_improvement"],
                size=n_rows,
            ),
            "verification_status": rng.choice(
                ["Verified", "Source Verified", "Not Verified"], size=n_rows
            ),
            "application_type": rng.choice(["Individual", "Joint App"], size=n_rows),
            "initial_list_status": rng.choice(["w", "f"], size=n_rows),
            "sub_grade": rng.choice(["A3", "B2", "C1", "D4"], size=n_rows),
        }
    )

    # Ensure high >= low for every row.
    df["fico_range_high"] = np.maximum(df["fico_range_high"], df["fico_range_low"] + 5)
    df["fico_range_high"] = np.clip(df["fico_range_high"], 300, 850)

    # Force both classes so model training is stable in CI.
    df["is_default"] = (rng.random(n_rows) < 0.2).astype(int)
    if df["is_default"].nunique() < 2:
        df.loc[df.index[0], "is_default"] = 0
        df.loc[df.index[1], "is_default"] = 1

    return df
