import xgboost as xgb
from credit_risk.models.base import BaseModel
from credit_risk.utils.config import xgb_config


class XGBoostModel(BaseModel):
    def __init__(self):
        self.model = xgb.XGBClassifier(**xgb_config.PARAMS)

    def train(self, X, y, eval_set=None):
        self.model.fit(
            X,
            y,
            eval_set=eval_set,
            verbose=False,
        )
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
