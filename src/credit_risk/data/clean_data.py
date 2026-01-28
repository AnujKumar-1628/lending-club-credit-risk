import pandas as pd
import numpy as np
from credit_risk.utils.logging import get_logger
from credit_risk.utils.config import data_config

logger = get_logger(__name__)


class DataCleaner:
    def __init__(self):
        self.missing_threshold = data_config.MISSING_THRESHOLD

    @staticmethod
    def _emp_length_to_num(val):
        if pd.isna(val):
            return np.nan
        if val == "< 1 year":
            return 0
        if val == "10+ years":
            return 10
        return int(val.split()[0])

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting data cleaning")
        df = df.copy()

        df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])]
        df["is_default"] = (df["loan_status"] == "Charged Off").astype(int)

        missing_ratio = df.isnull().mean()
        df = df.drop(
            columns=missing_ratio[missing_ratio > self.missing_threshold].index
        )

        keep_cols = [
            "addr_state",
            "is_default",
            "annual_inc",
            "application_type",
            "dti",
            "earliest_cr_line",
            "emp_length",
            "fico_range_high",
            "fico_range_low",
            "home_ownership",
            "initial_list_status",
            "installment",
            "int_rate",
            "issue_d",
            "loan_amnt",
            "mort_acc",
            "open_acc",
            "pub_rec",
            "pub_rec_bankruptcies",
            "purpose",
            "revol_bal",
            "revol_util",
            "sub_grade",
            "term",
            "title",
            "total_acc",
            "verification_status",
            "zip_code",
        ]
        df = df[keep_cols]

        df["emp_length_num"] = df["emp_length"].apply(self._emp_length_to_num)
        df["emp_length_missing"] = df["emp_length_num"].isna().astype(int)
        df["emp_length_num"] = df["emp_length_num"].fillna(
            df["emp_length_num"].median()
        )
        df = df.drop(columns=["emp_length", "title"])

        df = df.dropna(subset=["zip_code"])
        df["dti"] = df["dti"].fillna(df["dti"].median())

        df["revol_util_missing"] = df["revol_util"].isna().astype(int)
        df["revol_util"] = df["revol_util"].fillna(df["revol_util"].median())

        df["mort_acc_missing"] = df["mort_acc"].isna().astype(int)
        df["mort_acc"] = df["mort_acc"].fillna(df["mort_acc"].median())

        df["pub_rec_bankruptcies"] = df["pub_rec_bankruptcies"].fillna(0)

        logger.info(f"Cleaning done. Shape: {df.shape}")
        return df
