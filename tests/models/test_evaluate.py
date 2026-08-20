"""
Tests for model evaluation.
"""

from __future__ import annotations
from unittest.mock import Mock
import numpy as np
import pytest
from src.config.config import settings
from src.models.evaluate import evaluate_model
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score
)


def test_evaluate_model(
    mock_processed_data,
    mock_model,
):
    """
    Model should be evaluated successfully.
    """

    result = evaluate_model(
        processed=mock_processed_data,
        model=mock_model,
    )

    expected_probabilities = mock_model.predict_proba.return_value[:, 1]

    expected_predictions = (
        expected_probabilities >= settings.evaluation.threshold
    ).astype(int)

    assert np.array_equal(
        result.y_prob,
        expected_probabilities,
    )
    assert np.array_equal(
        result.y_pred,
        expected_predictions,
    )
    assert result.accuracy == accuracy_score(
        mock_processed_data.y_test,
        expected_predictions,
    )
    assert result.precision == precision_score(
        mock_processed_data.y_test,
        expected_predictions,
    )
    assert result.recall == recall_score(
        mock_processed_data.y_test,
        expected_predictions,
    )
    assert result.f1 == f1_score(
        mock_processed_data.y_test,
        expected_predictions,
    )
    assert result.roc_auc == roc_auc_score(
        mock_processed_data.y_test,
        expected_probabilities,
    )
    assert np.array_equal(
        result.confusion_matrix,
        confusion_matrix(
            mock_processed_data.y_test,
            expected_predictions,
        ),
    )


def test_evaluation_probability_shape(
    mock_processed_data,
    mock_model,
):
    """
    Probability output should have expected shape.
    """

    result = evaluate_model(
        processed=mock_processed_data,
        model=mock_model,
    )

    assert result.y_prob.shape == (2,)


def test_evaluation_prediction_shape(
    mock_processed_data,
    mock_model,
):
    """
    Prediction output should have expected shape.
    """

    result = evaluate_model(
        processed=mock_processed_data,
        model=mock_model,
    )

    assert result.y_pred.shape == (2,)


def test_evaluation_predictions_binary(
    mock_processed_data,
    mock_model,
):
    """
    Predictions should be binary.
    """

    result = evaluate_model(
        processed=mock_processed_data,
        model=mock_model,
    )

    assert set(result.y_pred.tolist()) <= {0, 1}


def test_evaluate_model_without_predict_proba(
    mock_processed_data,
):
    """
    Models without predict_proba should raise AttributeError.
    """

    class DummyModel:
        pass

    with pytest.raises(AttributeError):

        evaluate_model(
            processed=mock_processed_data,
            model=DummyModel(),
        )


def test_predict_proba_called(
    mock_processed_data,
    mock_model,
):
    """
    Model predict_proba should be called.
    """

    evaluate_model(
        processed=mock_processed_data,
        model=mock_model,
    )

    mock_model.predict_proba.assert_called_once_with(
        mock_processed_data.X_test_processed,
    )