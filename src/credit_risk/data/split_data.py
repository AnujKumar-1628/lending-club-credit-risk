import pandas as pd
from credit_risk.utils.logging import get_logger
from credit_risk.utils.config import split_config, data_config

logger = get_logger(__name__)


class DataSplitter:
    def split(self, df: pd.DataFrame):
        df = df.copy()
        df[data_config.DATE_COL] = pd.to_datetime(
            df[data_config.DATE_COL],
            format="%b-%Y",
            errors="coerce",
        )

        df = df.sort_values(data_config.DATE_COL)

        n = len(df)
        train_end = int(n * split_config.TRAIN_FRAC)
        val_end = int(n * (split_config.TRAIN_FRAC + split_config.VAL_FRAC))

        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]

        logger.info(
            f"Split sizes → train={len(train)}, val={len(val)}, test={len(test)}"
        )
        return train, val, test
