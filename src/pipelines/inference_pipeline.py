"""
Inference pipeline.
"""

from __future__ import annotations
import pandas as pd
from src.models.predict import (
    predict,
    predict_proba
)
from src.persistence.load import load_artifacts
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_inference(
    X: pd.DataFrame,
    *,
    return_probabilities: bool = False,
):
    """
    Run the end to end inference pipeline.
    """

    logger.info("Starting inference pipeline.")

    artifacts = load_artifacts()

    if return_probabilities:
        predictions = predict_proba(
            model=artifacts.model,
            preprocessor= artifacts.preprocessor,
            X= X
        )
    else:
        predictions = predict(
            model=artifacts.model,
            preprocessor= artifacts.preprocessor,
            X=X
        )

    logger.info("Inference pipeline completed successfully.")

    return predictions