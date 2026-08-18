"""
Generate all visualization for a trained model.
"""

from __future__ import annotations
from matplotlib.figure import Figure
from src.core import EvaluationResult, ExplainabilityResult
from .confusion_matrix import create_confusion_matrix
from .shap_summary import create_shap_summary
from .feature_importance import create_feature_importance
from .precision_recall_curve import create_precision_recall_curve
from .roc_curve import create_roc_curve

def generate_visualization(
        evaluation: EvaluationResult,
        explainability: ExplainabilityResult,
        model,
        feature_names: list[str]
) -> dict[str, Figure]:
    """
    Generate all figures for a trained model.
    """

    return {
        "roc_curve": create_roc_curve(evaluation),
        "precision_recall_curve": create_precision_recall_curve(evaluation),
        "confusion_matrix": create_confusion_matrix(evaluation),
        "feature_importance": create_feature_importance(
            model= model,
            feature_names= feature_names
        ),
        "shap_summary": create_shap_summary(explainability)
    }