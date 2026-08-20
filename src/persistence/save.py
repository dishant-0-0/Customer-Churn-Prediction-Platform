"""
Persistence utilities for deployment artifacts.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.config.config import settings
from src.config.paths import ARTIFACTS_DIR, CONFIG_DIR
from src.core import EvaluationResult, InferenceArtifacts
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_experiment_dir() -> Path:
    """
    Return the directory for the current experiment.
    """

    experiment_name = (
        f"{settings.training.model.name}_{settings.artifacts.version}"
    )

    return ARTIFACTS_DIR / experiment_name


def get_experiment_paths() -> dict[str, Path]:
    """
    Return artifact directories for the current experiment.
    """

    experiment_dir = get_experiment_dir()

    return {
        "experiment": experiment_dir,
        "model": experiment_dir / "model",
        "metrics": experiment_dir / "metrics",
        "figures": experiment_dir / "figures",
        "reports": experiment_dir / "reports",
    }


def _save_json(
    data: dict | list,
    output_path: Path,
) -> None:
    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
        )


def save_artifacts(artifacts: InferenceArtifacts, force: bool = False) -> Path:
    """
    Save deployment artifacts.
    """

    logger.info("Saving artifacts.")

    directories = get_experiment_paths()
    experiment_dir = directories["experiment"]

    if experiment_dir.exists():
        if force:
            logger.warning("Overwriting existing experiment.")
            shutil.rmtree(experiment_dir)
        else:
            raise FileExistsError("Experiment already exists.")

    for directory in directories.values():
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    joblib.dump(
        artifacts.model,
        directories["model"] / "model.joblib",
    )

    joblib.dump(
        artifacts.preprocessor, directories["model"] / "preprocessor.joblib"
    )

    _save_json(
        artifacts.feature_names,
        directories["model"] / "feature_names.json",
    )

    _save_json(
        {
            "high_value_threshold": artifacts.high_value_threshold,
        },
        directories["model"] / "metadata.json",
    )

    shutil.copy2(
        CONFIG_DIR / settings.files.config_file,
        directories["model"] / settings.files.config_file,
    )

    logger.info(f"Artifacts saved to '{experiment_dir}'.")

    return experiment_dir


def save_metrics(evaluation: EvaluationResult) -> Path:
    """
    Save evaluation metrics as JSON.
    """

    directories = get_experiment_paths()
    output_path = directories["metrics"] / "metrics.json"

    metrics = {
        "accuracy": evaluation.accuracy,
        "precision": evaluation.precision,
        "recall": evaluation.recall,
        "f1": evaluation.f1,
        "roc_auc": evaluation.roc_auc,
        "average_precision": evaluation.average_precision,
    }

    _save_json(metrics, output_path)

    logger.info(f"Saved metrics to '{output_path}'.")

    return output_path


def save_figure(
    figure: Figure,
    output_path: Path,
) -> Path:
    """
    Save a matplotlib figure.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    logger.info(f"Saved figure: {output_path}")

    return output_path


def save_figures(
    figures: dict[str, Figure],
) -> list[Path]:
    """
    Save all generated figures for the current experiment.
    """

    paths = get_experiment_paths()

    saved_paths: list[Path] = []

    for name, figure in figures.items():
        output_path = paths["figures"] / f"{name}.png"

        save_figure(
            figure,
            output_path,
        )

        saved_paths.append(output_path)

    return saved_paths
