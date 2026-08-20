"""
Tests for feature importance visualization.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure
import numpy as np
import pytest

from src.visualization.feature_importance import (
    create_feature_importance,
)


class DummyModel:
    """
    Dummy model exposing feature_importances_.
    """

    def __init__(
        self,
        importances,
    ):
        self.feature_importances_ = np.asarray(
            importances,
            dtype=float,
        )


def test_create_feature_importance_success():
    """
    Feature importance figure should be created.
    """

    model = DummyModel(
        [
            0.5,
            0.3,
            0.2,
        ]
    )

    feature_names = [
        "A",
        "B",
        "C",
    ]

    figure = create_feature_importance(
        model=model,
        feature_names=feature_names,
    )

    assert isinstance(
        figure,
        Figure,
    )


def test_create_feature_importance_missing_attribute():
    """
    Model without feature_importances_ should raise AttributeError.
    """

    class DummyModel:
        pass

    with pytest.raises(
        AttributeError,
        match="feature_importances_",
    ):
        create_feature_importance(
            model=DummyModel(),
            feature_names=["A"],
        )


def test_create_feature_importance_invalid_top_n():
    """
    top_n must be at least one.
    """

    model = DummyModel(
        [
            0.5,
            0.5,
        ]
    )

    with pytest.raises(
        ValueError,
        match="top_n",
    ):
        create_feature_importance(
            model=model,
            feature_names=[
                "A",
                "B",
            ],
            top_n=0,
        )


def test_create_feature_importance_feature_length_mismatch():
    """
    Number of feature names must match importances.
    """

    model = DummyModel(
        [
            0.6,
            0.4,
        ]
    )

    with pytest.raises(
        ValueError,
        match="same length",
    ):
        create_feature_importance(
            model=model,
            feature_names=[
                "A",
            ],
        )


def test_create_feature_importance_top_n():
    """
    Only the requested number of features should be plotted.
    """

    model = DummyModel(
        [
            0.40,
            0.30,
            0.20,
            0.10,
        ]
    )

    figure = create_feature_importance(
        model=model,
        feature_names=[
            "A",
            "B",
            "C",
            "D",
        ],
        top_n=2,
    )

    axes = figure.axes[0]

    labels = [
        tick.get_text()
        for tick in axes.get_yticklabels()
        if tick.get_text()
    ]

    assert len(labels) == 2


def test_create_feature_importance_sorted():
    """
    Features should be sorted by importance.
    """

    model = DummyModel(
        [
            0.10,
            0.80,
            0.30,
        ]
    )

    figure = create_feature_importance(
        model=model,
        feature_names=[
            "Feature A",
            "Feature B",
            "Feature C",
        ],
    )

    axes = figure.axes[0]

    labels = [
        tick.get_text()
        for tick in axes.get_yticklabels()
        if tick.get_text()
    ]

    assert labels == [
        "Feature B",
        "Feature C",
        "Feature A",
    ]