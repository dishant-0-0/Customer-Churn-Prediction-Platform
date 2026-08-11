"""
ROC Curve visualization.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from src.config.config import settings
from src.core import EvaluationResult
from src.visualization._utils import (
    create_figure,
    finalize_figure
)

def create_roc_curve(
    evaluation: EvaluationResult
) -> Figure:
    """
    Create a ROC Curve figure.
    """

    fig, ax = create_figure()

    ax.plot(
        evaluation.fpr,
        evaluation.tpr,
        color= settings.visualization.roc_curve.color,
        linewidth = settings.visualization.roc_curve.linewidth,
        label= f"AUC = {evaluation.roc_auc:.3f}"
    )

    ax.plot(
        [0,1],
        [0,1],
        linestyle= "--",
        color= "gray",
        linewidth = 1,
        label= "Random Classifier",
    )

    ax.set_title(
        "Receiver Operating Characteristic",
        fontsize= settings.visualization.font.title_size
    )

    ax.set_xlabel(
        "False Positive Rate",
        fontsize= settings.visualization.font.label_size,
    )

    ax.set_ylabel(
        "True Positive Rate",
        fontsize= settings.visualization.font.label_size,
    )

    ax.tick_params(
        labelsize= settings.visualization.font.tick_size
    )

    ax.legend()

    return finalize_figure(fig=fig)