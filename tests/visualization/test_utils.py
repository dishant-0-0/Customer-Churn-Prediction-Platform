"""
Tests for visualization utility functions.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.visualization._utils import (
    create_figure,
    finalize_figure,
)


def test_create_figure():
    """
    create_figure should return a Figure and Axes.
    """

    figure, axes = create_figure()

    assert isinstance(
        figure,
        Figure,
    )

    assert isinstance(
        axes,
        Axes,
    )

    assert axes.figure is figure


def test_finalize_figure_returns_same_instance():
    """
    finalize_figure should return the same Figure instance.
    """

    figure, _ = create_figure()

    returned = finalize_figure(
        figure,
    )

    assert returned is figure


def test_finalize_figure_after_plot():
    """
    finalize_figure should work after plotting.
    """

    figure, axes = create_figure()

    axes.plot(
        [1, 2, 3],
        [4, 5, 6],
    )

    returned = finalize_figure(
        figure,
    )

    assert returned is figure

    assert len(returned.axes) == 1


def test_create_multiple_figures():
    """
    Each call to create_figure should create new objects.
    """

    figure1, axes1 = create_figure()

    figure2, axes2 = create_figure()

    assert figure1 is not figure2

    assert axes1 is not axes2
