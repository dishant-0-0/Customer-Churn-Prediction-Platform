"""
Precision-Recall curve visualization.
"""

from __future__ import annotations

from matplotlib.figure import Figure

from src.config.config import settings
from src.core import EvaluationResult
from src.visualization._utils import create_figure, finalize_figure


def create_precision_recall_curve(
    evaluation: EvaluationResult,
) -> Figure:
    """
    Create a Precision-Recall curve.
    """

    fig, ax = create_figure()

    ax.plot(
        evaluation.recall_curve,
        evaluation.precision_curve,
        color=settings.visualization.roc_curve.color,
        linewidth=settings.visualization.roc_curve.linewidth,
        label=f"AP = {evaluation.average_precision:.3f}",
    )

    ax.set_title(
        "Precision-Recall Curve",
        fontsize=settings.visualization.font.title_size,
    )

    ax.set_xlabel("Recall", fontsize=settings.visualization.font.label_size)

    ax.set_ylabel("Precision", fontsize=settings.visualization.font.label_size)

    ax.tick_params(labelsize=settings.visualization.font.tick_size)

    ax.legend()

    return finalize_figure(fig)
