"""
Tests for the inference pipeline.
"""

from __future__ import annotations
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.config.config import settings
from src.core import PredictionResult
from src.pipelines.inference_pipeline import (
    run_inference_pipeline,
)


@patch("src.pipelines.inference_pipeline.predict_proba")
@patch("src.pipelines.inference_pipeline.feature_engineering_pipeline")
def test_run_inference_pipeline(
    mock_feature_engineering,
    mock_predict_proba,
    sample_prediction_dataframe,
    mock_inference_artifacts,
):
    """
    End-to-end inference pipeline should orchestrate
    feature engineering, prediction and thresholding.
    """

    engineered = sample_prediction_dataframe.copy()

    mock_feature_engineering.return_value = engineered

    probabilities = np.array(
        [
            0.8,
            0.3,
        ]
    )

    mock_predict_proba.return_value = probabilities

    result = run_inference_pipeline(
        X=sample_prediction_dataframe,
        artifacts=mock_inference_artifacts,
    )

    assert isinstance(
        result,
        PredictionResult,
    )

    mock_feature_engineering.assert_called_once_with(
        df=sample_prediction_dataframe,
        high_value_threshold=(
            mock_inference_artifacts.high_value_threshold
        ),
    )

    mock_predict_proba.assert_called_once_with(
        model=mock_inference_artifacts.model,
        preprocessor=mock_inference_artifacts.preprocessor,
        X=engineered,
    )


    np.testing.assert_array_equal(
        result.probabilities,
        probabilities,
    )

    expected_predictions = (
        probabilities >= settings.evaluation.threshold
    ).astype(int)

    np.testing.assert_array_equal(
        result.predictions,
        expected_predictions,
    )