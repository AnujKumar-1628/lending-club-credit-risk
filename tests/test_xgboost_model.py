from credit_risk.models.xgboost_model import XGBoostModel
from credit_risk.features.build_features import FeatureBuilder
from credit_risk.data.split_data import DataSplitter


def test_xgboost_model_train_predict(sample_cleaned_df):
    df = sample_cleaned_df.head(5000)

    splitter = DataSplitter()
    train_df, val_df, _ = splitter.split(df)

    fb = FeatureBuilder()
    X_train, y_train = fb.build_features(train_df, fit=True)
    X_val, y_val = fb.build_features(val_df, fit=False)

    model = XGBoostModel()
    model.train(X_train, y_train)

    preds = model.predict_proba(X_val)

    assert len(preds) == len(X_val)
