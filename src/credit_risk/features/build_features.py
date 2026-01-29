import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from credit_risk.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureBuilder:
    def __init__(self):
        self.preprocessor = None

        self.num_features = [
            "annual_inc",
            "dti",
            "installment",
            "int_rate",
            "loan_amnt",
            "revol_bal",
            "revol_util",
            "total_acc",
            "open_acc",
            "mort_acc",
            "emp_length_num",
            "fico_avg",
            "issue_year",
            "issue_month",
            "earliest_cr_year",
        ]

        self.binary_features = [
            "emp_length_missing",
            "revol_util_missing",
            "mort_acc_missing",
            "pub_rec",
            "pub_rec_bankruptcies",
        ]

        self.cat_features = [
            "addr_state",
            "application_type",
            "home_ownership",
            "initial_list_status",
            "purpose",
            "sub_grade",
            "term",
            "verification_status",
        ]

    def _add_core_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
        df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], errors="coerce")

        df["issue_year"] = df["issue_d"].dt.year
        df["issue_month"] = df["issue_d"].dt.month
        df["earliest_cr_year"] = df["earliest_cr_line"].dt.year

        df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

        return df.drop(
            columns=["issue_d", "earliest_cr_line", "fico_range_low", "fico_range_high"]
        )

    def build_features(self, df: pd.DataFrame, fit: bool = True):
        logger.info("Building features")

        y = df["is_default"] if "is_default" in df.columns else None
        X = df.drop(columns=["is_default"]) if y is not None else df

        X = self._add_core_features(X)

        if fit:
            self.preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num",
                        Pipeline(
                            [
                                ("imputer", SimpleImputer(strategy="median")),
                                ("scaler", StandardScaler()),
                            ]
                        ),
                        self.num_features,
                    ),
                    (
                        "bin",
                        SimpleImputer(strategy="most_frequent"),
                        self.binary_features,
                    ),
                    (
                        "cat",
                        Pipeline(
                            [
                                ("imputer", SimpleImputer(strategy="most_frequent")),
                                (
                                    "onehot",
                                    OneHotEncoder(
                                        handle_unknown="ignore", sparse_output=False
                                    ),
                                ),
                            ]
                        ),
                        self.cat_features,
                    ),
                ]
            )
            X = self.preprocessor.fit_transform(X)
        else:
            X = self.preprocessor.transform(X)

        return X, y
