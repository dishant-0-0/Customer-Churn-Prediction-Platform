"""Reusable charts for interpreting trained models."""

from collections.abc import Sequence
from src.visualization._utils import(
    create_figure,
    finalize_figure
)
import matplotlib.pyplot as plt
from src.utils.feature_names import clean_feature_names


def create_feature_importance(model, feature_names: Sequence[str], top_n: int = 20):
    """Plot the most important features exposed by a fitted tree-based model.

    Returns the matplotlib axes containing the chart.
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            "The model must expose a 'feature_importances_' attribute."
        )
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    feature_importances = model.feature_importances_
    if len(feature_names) != len(feature_importances):
        raise ValueError(
            "feature_names and model.feature_importances_ must have the same length."
        )

    ranked_features = sorted(
        zip(feature_names, feature_importances),
        key=lambda feature: feature[1],
        reverse=True,
    )[:top_n]

    names, importances = zip(*ranked_features)

    fig, axes = create_figure()

    axes.barh(names, importances)
    axes.invert_yaxis()
    axes.set_xlabel("Feature Importance")
    axes.set_ylabel("Feature")
    axes.set_title(f"Top {len(ranked_features)} Important Features")

    return finalize_figure(fig)
