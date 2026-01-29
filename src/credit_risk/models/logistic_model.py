from sklearn.linear_model import SGDClassifier
from credit_risk.models.base import BaseModel
from credit_risk.utils.config import model_config


class LogisticSGDModel(BaseModel):
    def __init__(self):
        self.model = SGDClassifier(**model_config.SGD_LOGISTIC_PARAMS)

    def train(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
