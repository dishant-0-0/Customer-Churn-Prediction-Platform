"""
Shared visualization utilities.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from src.config.config import settings

def create_figure() -> tuple[Figure, Axes]:
    """
    Create a standardized matplotlib figure.
    """
    plt.style.use(settings.visualization.style)

    fig,ax = plt.subplots(
        figsize=(
            settings.visualization.figsize.width,
            settings.visualization.figsize.height
        ),
        dpi = settings.visualization.dpi
    )

    return fig, ax

def finalize_figure(
        fig:Figure
) -> Figure:
    """
    Apply final formatting before returning a figure.
    """

    fig.tight_layout()

    return fig