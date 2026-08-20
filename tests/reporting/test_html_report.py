"""
Tests for HTML report generation.
"""

from __future__ import annotations

from pathlib import Path

from src.reporting.html_report import (
    _build_experiment_section,
    _build_figures,
    _build_footer,
    _build_header,
    _build_metrics_table,
    generate_html_report,
)


def test_build_header():
    """
    HTML header should contain the document structure.
    """

    html = _build_header()

    assert "<!DOCTYPE html>" in html
    assert "<html>" in html
    assert "<head>" in html
    assert "<body>" in html
    assert "<style>" in html


def test_build_experiment_section():
    """
    Experiment section should contain metadata.
    """

    html = _build_experiment_section()

    assert "Customer Churn Prediction Report" in html
    assert "Experiment" in html
    assert "Model" in html
    assert "Version" in html
    assert "Threshold" in html


def test_build_metrics_table(
    mock_training_result,
):
    """
    Metrics table should contain formatted metrics.
    """

    html = _build_metrics_table(
        mock_training_result,
    )

    assert "Accuracy" in html
    assert "Precision" in html
    assert "Recall" in html
    assert "F1 Score" in html
    assert "ROC AUC" in html
    assert "Average Precision" in html

    assert f"{mock_training_result.evaluation.accuracy:.4f}" in html

    assert f"{mock_training_result.evaluation.precision:.4f}" in html


def test_build_figures():
    """
    HTML should reference every visualization.
    """

    html = _build_figures()

    expected = [
        "roc_curve",
        "precision_recall_curve",
        "confusion_matrix",
        "feature_importance",
        "shap_summary",
    ]

    for figure in expected:
        assert f"{figure}.png" in html

    assert html.count("<img") == len(expected)


def test_build_footer():
    """
    Footer should contain closing HTML.
    """

    html = _build_footer()

    assert "Customer Churn ML Framework" in html
    assert "</body>" in html
    assert "</html>" in html


def test_generate_html_report(
    mock_training_result,
    tmp_path,
    monkeypatch,
):
    """
    HTML report should be generated successfully.
    """

    report_dir = tmp_path / "reports"
    report_dir.mkdir()

    monkeypatch.setattr(
        "src.reporting.html_report.get_experiment_paths",
        lambda: {
            "reports": report_dir,
        },
    )

    report_path = generate_html_report(
        mock_training_result,
    )

    assert isinstance(
        report_path,
        Path,
    )

    assert report_path.exists()

    html = report_path.read_text(
        encoding="utf-8",
    )

    assert "Customer Churn Prediction Report" in html
    assert "Performance" in html
    assert "Visualizations" in html

    assert f"{mock_training_result.evaluation.accuracy:.4f}" in html

    assert f"{mock_training_result.evaluation.precision:.4f}" in html
