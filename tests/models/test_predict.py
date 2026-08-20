"""
Tests for prediction utilities.
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from src.config.config import settings
from src.models.predict import (
    _prepare_features,
    predict,
    predict_proba,
)


def test_prepare_features(
    sample_prediction_dataframe,
    mock_preprocessor,
):
    """
    Features should be transformed using the preprocessor.
    """

    result = _prepare_features(
        preprocessor=mock_preprocessor,
        X=sample_prediction_dataframe,
    )

    assert np.array_equal(
        result,
        mock_preprocessor.transform.return_value,
    )

    mock_preprocessor.transform.assert_called_once_with(
        sample_prediction_dataframe,
    )


def test_predict_proba(
    sample_prediction_dataframe,
    mock_preprocessor,
    mock_model,
):
    """
    Prediction probabilities should be returned.
    """

    result = predict_proba(
        model=mock_model,
        preprocessor=mock_preprocessor,
        X=sample_prediction_dataframe,
    )

    expected = np.array([0.7, 0.2])

    assert np.array_equal(
        result,
        expected,
    )

    mock_preprocessor.transform.assert_called_once_with(
        sample_prediction_dataframe,
    )
    mock_model.predict_proba.assert_called_once_with(
        mock_preprocessor.transform.return_value,
    )


def test_predict(
    sample_prediction_dataframe,
    mock_preprocessor,
    mock_model,
):
    """
    Predictions should use the configured threshold.
    """

    result = predict(
        model=mock_model,
        preprocessor=mock_preprocessor,
        X=sample_prediction_dataframe,
    )

    expected = np.array(
        [
            int(0.7 >= settings.evaluation.threshold),
            int(0.2 >= settings.evaluation.threshold),
        ]
    )

    assert np.array_equal(
        result,
        expected,
    )


def test_predict_binary(
    sample_prediction_dataframe,
    mock_preprocessor,
    mock_model,
):
    """
    Predictions should be binary.
    """

    result = predict(
        model=mock_model,
        preprocessor=mock_preprocessor,
        X=sample_prediction_dataframe,
    )

    assert set(result.tolist()) <= {0, 1}


def test_predict_output_shape(
    sample_prediction_dataframe,
    mock_preprocessor,
    mock_model,
):
    """
    Predictions should have the expected shape.
    """

    result = predict(
        model=mock_model,
        preprocessor=mock_preprocessor,
        X=sample_prediction_dataframe,
    )

    assert result.shape == (2,)


def test_predict_proba_missing_method(
    sample_prediction_dataframe,
    mock_preprocessor,
):
    """
    Models without predict_proba should raise AttributeError.
    """

    class DummyModel:
        pass

    with pytest.raises(AttributeError):

        predict_proba(
            model=DummyModel(),
            preprocessor=mock_preprocessor,
            X=sample_prediction_dataframe,
        )