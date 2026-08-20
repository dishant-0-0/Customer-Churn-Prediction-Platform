"""
Tests for model registry.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.models.registry import (
    MODEL_REGISTRY,
    get_model,
)


def test_model_registry_contains_supported_models():
    """
    Registry should contain all supported models.
    """

    assert MODEL_REGISTRY == {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "xgboost": XGBClassifier,
    }


@patch("src.models.registry.settings")
def test_get_model_logistic_regression(
    mock_settings,
):
    """
    Logistic Regression model should be created from configuration.
    """

    mock_settings.training.model.name = "logistic_regression"
    mock_settings.training.model.params = {
        "random_state": 42,
    }

    model = get_model()

    assert isinstance(
        model,
        LogisticRegression,
    )

    assert model.random_state == 42


@patch("src.models.registry.settings")
def test_get_model_random_forest(
    mock_settings,
):
    """
    Random Forest model should be created from configuration.
    """

    mock_settings.training.model.name = "random_forest"
    mock_settings.training.model.params = {
        "n_estimators": 10,
        "max_depth": 3,
        "random_state": 42,
    }

    model = get_model()

    assert isinstance(
        model,
        RandomForestClassifier,
    )

    assert model.n_estimators == 10
    assert model.max_depth == 3
    assert model.random_state == 42


@patch("src.models.registry.settings")
def test_get_model_xgboost(
    mock_settings,
):
    """
    XGBoost model should be created from configuration.
    """

    mock_settings.training.model.name = "xgboost"
    mock_settings.training.model.params = {
        "n_estimators": 20,
        "max_depth": 4,
        "learning_rate": 0.1,
    }

    model = get_model()

    assert isinstance(
        model,
        XGBClassifier,
    )

    assert model.n_estimators == 20
    assert model.max_depth == 4
    assert model.learning_rate == 0.1


@patch("src.models.registry.settings")
def test_get_model_unknown_model(
    mock_settings,
):
    """
    Unknown model names should raise ValueError.
    """

    mock_settings.training.model.name = "unknown_model"
    mock_settings.training.model.params = {}

    with pytest.raises(
        ValueError,
        match="Unknown model",
    ):
        get_model()
