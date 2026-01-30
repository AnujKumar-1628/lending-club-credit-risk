from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    RAW_FILENAME: str = "accepted_2007_to_2018Q4.csv"
    CLEANED_FILENAME: str = "cleaned_data.parquet"
    TARGET_COL: str = "is_default"
    DATE_COL: str = "issue_d"
    MISSING_THRESHOLD: float = 0.30


@dataclass(frozen=True)
class SplitConfig:
    TRAIN_FRAC: float = 0.70
    VAL_FRAC: float = 0.15


data_config = DataConfig()
split_config = SplitConfig()


@dataclass(frozen=True)
class ModelConfig:
    SGD_LOGISTIC_PARAMS: dict = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "SGD_LOGISTIC_PARAMS",
            {
                "loss": "log_loss",
                "penalty": "l2",
                "alpha": 0.0001,
                "learning_rate": "optimal",
                "max_iter": 1000,
                "tol": 1e-3,
                "random_state": 42,
            },
        )


model_config = ModelConfig()


@dataclass(frozen=True)
class XGBoostConfig:
    PARAMS: dict = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "PARAMS",
            {
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "random_state": 42,
                "n_jobs": -1,
            },
        )


xgb_config = XGBoostConfig()
