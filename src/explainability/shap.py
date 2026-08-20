"""
SHAP explainability utilites.
"""

from __future__ import annotations

import pandas as pd
import shap

from src.core import ExplainabilityResult


def generate_explanation(model, X: pd.DataFrame) -> ExplainabilityResult:
    """
    Generate SHAP explanations for a fitted model.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame with feature names.")

    explainer = shap.Explainer(model)

    explanation = explainer(X)

    return ExplainabilityResult(explanation=explanation)
