"""
Tests for visualization generation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from matplotlib.figure import Figure

from src.visualization.generate import (
    generate_visualization,
)


@patch("src.visualization.generate.create_shap_summary")
@patch("src.visualization.generate.create_feature_importance")
@patch("src.visualization.generate.create_confusion_matrix")
@patch("src.visualization.generate.create_precision_recall_curve")
@patch("src.visualization.generate.create_roc_curve")
def test_generate_visualization(
    mock_roc,
    mock_pr,
    mock_cm,
    mock_feature_importance,
    mock_shap,
    mock_evaluation_result,
):
    """
    generate_visualization should orchestrate all visualization
    functions and return the generated figures.
    """

    roc = Figure()
    pr = Figure()
    cm = Figure()
    fi = Figure()
    shap = Figure()

    mock_roc.return_value = roc
    mock_pr.return_value = pr
    mock_cm.return_value = cm
    mock_feature_importance.return_value = fi
    mock_shap.return_value = shap

    model = MagicMock()
    explainability = MagicMock()

    feature_names = [
        "feature_1",
        "feature_2",
    ]

    result = generate_visualization(
        evaluation=mock_evaluation_result,
        explainability=explainability,
        model=model,
        feature_names=feature_names,
    )

    mock_roc.assert_called_once_with(
        mock_evaluation_result,
    )

    mock_pr.assert_called_once_with(
        mock_evaluation_result,
    )

    mock_cm.assert_called_once_with(
        mock_evaluation_result,
    )

    mock_feature_importance.assert_called_once_with(
        model=model,
        feature_names=feature_names,
    )

    mock_shap.assert_called_once_with(
        explainability,
    )

    assert result == {
        "roc_curve": roc,
        "precision_recall_curve": pr,
        "confusion_matrix": cm,
        "feature_importance": fi,
        "shap_summary": shap,
    }

    assert set(result.keys()) == {
        "roc_curve",
        "precision_recall_curve",
        "confusion_matrix",
        "feature_importance",
        "shap_summary",
    }

    assert all(
        isinstance(
            figure,
            Figure,
        )
        for figure in result.values()
    )