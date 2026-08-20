"""
Inference pipeline.
"""

from __future__ import annotations

import pandas as pd

from src.config.config import settings
from src.core import InferenceArtifacts, PredictionResult
from src.features.feature_engineering import feature_engineering_pipeline
from src.models.predict import predict_proba
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_inference_pipeline(
    X: pd.DataFrame,
    artifacts: InferenceArtifacts,
) -> PredictionResult:
    """
    Run the end to end inference pipeline.
    """

    logger.info("Starting inference pipeline.")

    logger.info("Applying feature engineering.")

    X = feature_engineering_pipeline(
        df=X,
        high_value_threshold=artifacts.high_value_threshold,
    )

    probabilities = predict_proba(
        model=artifacts.model, preprocessor=artifacts.preprocessor, X=X
    )

    logger.info("Applying decision threshold.")

    predictions = (probabilities >= settings.evaluation.threshold).astype(int)

    logger.info("Inference pipeline completed successfully.")

    return PredictionResult(
        predictions=predictions,
        probabilities=probabilities,
    )
