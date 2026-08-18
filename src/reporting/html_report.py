"""
Generate HTML experiment report.
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from src.config.config import settings
from src.core import TrainingResult
from src.persistence.save import get_experiment_paths
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_header() -> str:
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Training Report</title>

        <style>

        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            line-height: 1.5;
        }

        table {
            border-collapse: collapse;
            width: 400px;
        }

        th, td {
            border: 1px solid #cccccc;
            padding: 8px;
        }

        th {
            background-color: #f2f2f2;
        }

        img {
            width: 900px;
            margin-bottom: 30px;
            border: 1px solid #dddddd;
        }

        h1, h2 {
            color: #333333;
        }

        </style>

    </head>

    <body>
    """


def _build_experiment_section() -> str:
    return f"""
    <h1>Customer Churn Prediction Report</h1>

    <h2>Experiment</h2>

    <table>

        <tr><th>Model</th><td>{settings.training.model.name}</td></tr>

        <tr><th>Version</th><td>{settings.artifacts.version}</td></tr>

        <tr><th>Generated</th><td>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</td></tr>

        <tr><th>Threshold</th><td>{settings.evaluation.threshold}</td></tr>

    </table>
    """


def _build_metrics_table(
    training_result: TrainingResult,
) -> str:

    evaluation = training_result.evaluation

    return f"""
    <h2>Performance</h2>

    <table>

        <tr><th>Accuracy</th><td>{evaluation.accuracy:.4f}</td></tr>

        <tr><th>Precision</th><td>{evaluation.precision:.4f}</td></tr>

        <tr><th>Recall</th><td>{evaluation.recall:.4f}</td></tr>

        <tr><th>F1 Score</th><td>{evaluation.f1:.4f}</td></tr>

        <tr><th>ROC AUC</th><td>{evaluation.roc_auc:.4f}</td></tr>

        <tr><th>Average Precision</th><td>{evaluation.average_precision:.4f}</td></tr>

    </table>
    """


def _build_figures() -> str:

    figures = [
        "roc_curve",
        "precision_recall_curve",
        "confusion_matrix",
        "feature_importance",
        "shap_summary",
    ]

    html = "<h2>Visualizations</h2>"

    for figure in figures:

        html += f"""
        <h3>{figure.replace("_", " ").title()}</h3>

        <img
            src="../figures/{figure}.png"
            alt="{figure}"
        >
        """

    return html


def _build_footer() -> str:

    return """
    <hr>

    <p>
        Generated automatically by the
        Customer Churn ML Framework
    </p>

    </body>
    </html>
    """


def generate_html_report(
    training_result: TrainingResult
) -> Path:
    """
    Generate an HTML report for a training run.
    """

    logger.info("Generating HTML report.")

    paths = get_experiment_paths()

    report_path = (
        paths["reports"] /
        "report.html"
    )

    html = (
        _build_header()
        + _build_experiment_section()
        + _build_metrics_table(training_result)
        + _build_figures()
        + _build_footer()
    )

    logger.info(f"Writing HTML report to '{report_path}'.")

    report_path.write_text(
        html,
        encoding="utf-8"
    )

    logger.info("HTML report generated successfully.")

    return report_path