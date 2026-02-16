from credit_risk.features.build_features import FeatureBuilder
from credit_risk.data.split_data import DataSplitter


def test_feature_builder_consistency(sample_cleaned_df):
    df = sample_cleaned_df.head(10000)

    splitter = DataSplitter()
    train_df, val_df, _ = splitter.split(df)

    fb = FeatureBuilder()
    X_train, y_train = fb.build_features(train_df, fit=True)
    X_val, y_val = fb.build_features(val_df, fit=False)

    assert X_train.shape[1] == X_val.shape[1]
