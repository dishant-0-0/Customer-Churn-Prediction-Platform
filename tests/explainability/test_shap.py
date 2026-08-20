"""
Tests for SHAP explainability utilities.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.core import ExplainabilityResult
from src.explainability.shap import generate_explanation


def test_generate_explanation_invalid_input():
    """
    Non-DataFrame input should raise TypeError.
    """

    with pytest.raises(
        TypeError,
        match="X must be a pandas DataFrame",
    ):
        generate_explanation(
            model=MagicMock(),
            X=[[1, 2], [3, 4]],
        )


@patch("src.explainability.shap.shap.Explainer")
def test_generate_explanation(
    mock_explainer_class,
):
    """
    SHAP explanation should be generated successfully.
    """

    model = MagicMock()

    X = pd.DataFrame(
        {
            "feature_1": [1.0, 2.0],
            "feature_2": [3.0, 4.0],
        }
    )

    explanation = MagicMock()

    explainer = MagicMock()
    explainer.return_value = explanation

    mock_explainer_class.return_value = explainer

    result = generate_explanation(
        model=model,
        X=X,
    )

    mock_explainer_class.assert_called_once_with(
        model,
    )

    explainer.assert_called_once_with(
        X,
    )

    assert isinstance(
        result,
        ExplainabilityResult,
    )

    assert result.explanation is explanation


@patch("src.explainability.shap.shap.Explainer")
def test_generate_explanation_passes_dataframe(
    mock_explainer_class,
):
    """
    The original DataFrame should be passed to SHAP.
    """

    model = MagicMock()

    X = pd.DataFrame(
        {
            "A": [1, 2],
            "B": [3, 4],
        }
    )

    explainer = MagicMock()
    explainer.return_value = MagicMock()

    mock_explainer_class.return_value = explainer

    generate_explanation(
        model=model,
        X=X,
    )

    passed_df = explainer.call_args.args[0]

    pd.testing.assert_frame_equal(
        passed_df,
        X,
    )


@patch("src.explainability.shap.shap.Explainer")
def test_generate_explanation_returns_same_object(
    mock_explainer_class,
):
    """
    Wrapper should return the exact explanation object
    produced by SHAP.
    """

    explanation = object()

    explainer = MagicMock()
    explainer.return_value = explanation

    mock_explainer_class.return_value = explainer

    result = generate_explanation(
        model=MagicMock(),
        X=pd.DataFrame(
            {
                "A": [1],
            }
        ),
    )

    assert result.explanation is explanation
