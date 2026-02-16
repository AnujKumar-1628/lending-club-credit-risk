from credit_risk.data.split_data import DataSplitter


def test_time_based_split_order(sample_cleaned_df):
    df = sample_cleaned_df.head(10000)

    splitter = DataSplitter()
    train_df, val_df, test_df = splitter.split(df)

    assert train_df["issue_d"].max() <= val_df["issue_d"].min()
    assert val_df["issue_d"].max() <= test_df["issue_d"].min()
