"""
Tests for model registry.
"""

from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from src.models.registry import (
    MODEL_REGISTRY,
    get_model,
)


def test_model_registry_contains_supported_models():
    """
    Registry should contain all supported models.
    """

    assert set(MODEL_REGISTRY.keys()) == {
        "logistic_regression",
        "random_forest",
        "xgboost",
    }


@patch("src.models.registry.settings")
@patch("src.models.registry.MODEL_REGISTRY")
def test_get_model_success(
    mock_registry,
    mock_settings,
):
    """
    Configured model should be created successfully.
    """

    model = MagicMock()

    model_class = MagicMock(
        return_value=model,
    )

    mock_registry.__contains__.return_value = True
    mock_registry.__getitem__.return_value = model_class

    mock_settings.training.model.name = "dummy_model"

    mock_settings.training.model.params = {
        "n_estimators": 100,
        "max_depth": 6,
    }

    result = get_model()

    model_class.assert_called_once_with(
        n_estimators=100,
        max_depth=6,
    )

    assert result is model


@patch("src.models.registry.settings")
@patch("src.models.registry.MODEL_REGISTRY")
def test_get_model_unknown_model(
    mock_registry,
    mock_settings,
):
    """
    Unknown model names should raise ValueError.
    """

    mock_registry.__contains__.return_value = False
    mock_registry.keys.return_value = [
        "logistic_regression",
        "random_forest",
    ]

    mock_settings.training.model.name = "unknown_model"

    with pytest.raises(
        ValueError,
        match="Unknown model",
    ):
        get_model()


@patch("src.models.registry.settings")
@patch("src.models.registry.MODEL_REGISTRY")
def test_get_model_passes_parameters(
    mock_registry,
    mock_settings,
):
    """
    Constructor should receive configuration parameters.
    """

    model = MagicMock()

    model_class = MagicMock(
        return_value=model,
    )

    mock_registry.__contains__.return_value = True
    mock_registry.__getitem__.return_value = model_class

    params = {
        "learning_rate": 0.05,
        "max_depth": 4,
        "n_estimators": 200,
    }

    mock_settings.training.model.name = "xgboost"
    mock_settings.training.model.params = params

    get_model()

    model_class.assert_called_once_with(
        **params,
    )