"""
Module for creating ML model from configurations.
"""

from __future__ import annotations
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_REGISTRY: dict[str, type[BaseEstimator]] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
}

def get_model() -> BaseEstimator:
    """
    Create an untrained model from the project configuration.
    """

    model_name = settings.training.model.name

    logger.info("Creating model: %s", model_name)

    try:
        model_class = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        ) from exc

    model_class = MODEL_REGISTRY[model_name]

    model = model_class(
        **settings.training.model.params,
    )

    logger.info("Successfully created model: %s", model_name)

    return model