"""
Confusion matrix visualization.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from src.config.config import settings
from src.core import EvaluationResult

def create_confusion_matrix(
    evaluation: EvaluationResult
) -> Figure:
    """
    Create a confusion matrix figure.
    """

    plt.style.use(settings.visualization.style)

    fig, ax = plt.subplots(
        figsize=(
            settings.visualization.figsize.width,
            settings.visualization.figsize.height
        )
    )

    sns.heatmap(
        evaluation.confusion_matrix,
        annot=settings.visualization.confusion_matrix.annot,
        fmt="d",
        cmap=settings.visualization.confusion_matrix.cmap,
        cbar=False,
        ax=ax
    )

    ax.set_title(
        "Confusion Matrix",
        fontsize= settings.visualization.font.title_size
        )
    
    ax.set_xlabel(
        "Predicted Label",
        fontsize= settings.visualization.font.label_size
        )
    
    ax.set_ylabel(
        "True Label",
        fontsize= settings.visualization.font.label_size
        )

    ax.tick_params(
        labelsize = settings.visualization.font.tick_size
    )

    fig.tight_layout()

    return fig