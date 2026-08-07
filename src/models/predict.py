"""
Prediction utilities.
"""

from __future__ import annotations
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from src.utils.logger import get_logger

logger = get_logger(__name__)

def predict(
    model: BaseEstimator,
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    *,
    return_proba: bool = False,
):
    """
    Generate predictions for unseen data.
    """

    logger.info("Generating predictions using %s", model.__class__.__name__)

    X_processed = preprocessor.transform(X)

    if return_proba:
        if not hasattr(model, "predict_proba"):
            raise AttributeError(
                f"{model.__class__.__name__} does not implement 'predict_proba()'."
            )

        return model.predict_proba(
            X_processed,
        )[:1]

    return model.predict(X_processed)