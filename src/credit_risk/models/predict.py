def predict_scores(model, X):
    """
    Returns default probability scores.
    """
    return model.predict_proba(X)
