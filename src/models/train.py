"""
Model training utilities.
"""

from __future__ import annotations
from sklearn.base import BaseEstimator
from src.core import ProcessedData
from src.utils.logger import get_logger

logger = get_logger(__name__)

def train_model(
    processed: ProcessedData,
    model: BaseEstimator,
) -> BaseEstimator:
    """
    Train a machine learning model.
    """

    logger.info("Training model: %s", model.__class__.__name__)

    model.fit(
        processed.X_train_processed,
        processed.y_train
    )

    logger.info("Model training completed.")

    return model
