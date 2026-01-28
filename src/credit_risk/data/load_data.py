import pandas as pd

from credit_risk.utils.paths import raw_dir, processed_dir
from credit_risk.utils.config import data_config
from credit_risk.utils.logging import get_logger

logger = get_logger(__name__)


USE_COLS = [
    "addr_state",
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
    "loan_status",
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


def load_raw_data(chunk_size: int = 200_000) -> pd.DataFrame:
    """
    Load raw LendingClub CSV using chunked reading
    to avoid out-of-memory errors.
    """

    path = raw_dir / data_config.RAW_FILENAME
    if not path.exists():
        raise FileNotFoundError(path)

    logger.info(f"Loading raw CSV in chunks from {path}")
    logger.info(f"Chunk size: {chunk_size:,}")

    chunks = []

    for i, chunk in enumerate(
        pd.read_csv(
            path,
            usecols=USE_COLS,
            chunksize=chunk_size,
            low_memory=False,
        ),
        start=1,
    ):
        logger.info(f"Loaded chunk {i} with shape {chunk.shape}")
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"Final raw dataframe shape: {df.shape}")

    return df


def load_cleaned_data() -> pd.DataFrame:
    """
    Load cleaned parquet data.

    """

    path = processed_dir / data_config.CLEANED_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at {path}. " "Run scripts/run_split.py first."
        )

    logger.info(f"Loading cleaned data from {path}")
    df = pd.read_parquet(path)
    logger.info(f"Cleaned data shape: {df.shape}")

    return df
