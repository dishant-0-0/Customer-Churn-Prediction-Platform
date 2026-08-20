"""
Tests for model training utilities.
"""

from __future__ import annotations
from unittest.mock import MagicMock
from src.models.train import train_model


def test_train_model(
    mock_processed_data,
):
    """
    Model should be trained using the processed data.
    """

    model = MagicMock()

    returned = train_model(
        processed=mock_processed_data,
        model=model,
    )

    model.fit.assert_called_once_with(
        mock_processed_data.X_train_processed,
        mock_processed_data.y_train,
    )

    assert returned is model


def test_train_model_returns_same_instance(
    mock_processed_data,
):
    """
    train_model should return the same fitted model instance.
    """

    model = MagicMock()

    returned = train_model(
        processed=mock_processed_data,
        model=model,
    )

    assert returned is model