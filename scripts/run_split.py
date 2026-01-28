from credit_risk.data.load_data import load_raw_data
from credit_risk.data.clean_data import DataCleaner
from credit_risk.data.split_data import DataSplitter
from credit_risk.utils.paths import processed_dir, samples_dir
from credit_risk.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    # Ensure directories exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data (chunked)
    df_raw = load_raw_data()
    logger.info(f"Raw data loaded: {df_raw.shape}")

    # 2. Clean data
    cleaner = DataCleaner()
    df_clean = cleaner.clean(df_raw)
    logger.info(f"Cleaned data shape: {df_clean.shape}")

    # 3. SAVE cleaned parquet (IMPORTANT)
    cleaned_path = processed_dir / "cleaned_data.parquet"
    df_clean.to_parquet(cleaned_path, index=False)
    logger.info(f"Saved cleaned data to {cleaned_path}")

    # 4. Time-based split
    splitter = DataSplitter()
    train_df, val_df, test_df = splitter.split(df_clean)

    # 5. Save splits
    train_df.to_parquet(samples_dir / "train.parquet", index=False)
    val_df.to_parquet(samples_dir / "val.parquet", index=False)
    test_df.to_parquet(samples_dir / "test.parquet", index=False)

    logger.info("Data pipeline completed successfully")


if __name__ == "__main__":
    main()
