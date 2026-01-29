from credit_risk.utils.logging import get_logger

logger = get_logger(__name__)


def train_model(model, X_train, y_train, X_val=None, y_val=None):
    logger.info("Starting model training")

    if X_val is not None and y_val is not None:
        model.train(X_train, y_train, eval_set=[(X_val, y_val)])
    else:
        model.train(X_train, y_train)

    logger.info("Training completed")
    return model
