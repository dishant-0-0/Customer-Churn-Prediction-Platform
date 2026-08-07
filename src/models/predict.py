"""
Prediction utilities.
"""

from __future__ import annotations
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from src.utils.logger import get_logger

logger = get_logger(__name__)

def _prepare_features(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
):
    """
    Transform raw features using the fitted preprocessor.
    """

    return preprocessor.transform(X)

def predict(
    model: BaseEstimator,
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
):
    """
    Generate predictions for unseen data.
    """

    logger.info("Generating predictions using %s", model.__class__.__name__)

    X_processed = _prepare_features(
        preprocessor=preprocessor,
        X= X
    )

    return model.predict(X_processed)

def predict_proba(
    model: BaseEstimator,
    preprocessor: ColumnTransformer,
    X: pd.DataFrame
):
    """
    Generate prediction probabilities for unseen data.
    """

    logger.info("Generating prediction probabilities using %s", model.__class__.__name__)

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            f"{model.__class__.__name__} "
            "does not implement 'predict_proba()'."
        )

    X_processed = _prepare_features(
        preprocessor,
        X
    )

    return model.predict_proba(
        X_processed
    )[:,1]