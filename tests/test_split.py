from credit_risk.data.split_data import DataSplitter
from credit_risk.data.load_data import load_cleaned_data


def test_time_based_split_order():
    df = load_cleaned_data().head(10000)

    splitter = DataSplitter()
    train_df, val_df, test_df = splitter.split(df)

    assert train_df["issue_d"].max() <= val_df["issue_d"].min()
    assert val_df["issue_d"].max() <= test_df["issue_d"].min()
