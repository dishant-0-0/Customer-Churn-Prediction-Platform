"""
SHAP summary visualization.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import shap
from matplotlib.figure import Figure
from src.core import ExplainabilityResult
from src.visualization._utils import finalize_figure


def create_shap_summary(
    explanation: ExplainabilityResult
) -> Figure:
    """
    Create a SHAP beeswam summary plot.
    """

    shap.plots.beeswarm(
        explanation.explanation,
        show= False
    )

    fig = plt.gcf()

    return finalize_figure(fig)