"""
Persistence utilities for deployment artifacts.
"""

from __future__ import annotations
import json, shutil, joblib
from pathlib import Path
from src.config.config import settings
from src.config.paths import (
    ARTIFACTS_DIR,
    CONFIG_DIR
)
from src.persistence import InferenceArtifacts
from src.utils.logger import get_logger

logger = get_logger(__name__)

def save_artifacts(
    artifacts: InferenceArtifacts,
    force: bool = False
) -> Path:
    """
    Save deployment artifacts.
    """

    logger.info("Saving artifacts.")

    experiment_name = f"{settings.training.model.name}_{settings.artifacts.version}"

    experiment_dir = ARTIFACTS_DIR / experiment_name

    if experiment_dir.exists():
        if force:
            logger.warning(
                f"Overwriting existing experiment: '{experiment_name}'"
            )
            shutil.rmtree(experiment_dir)
        else: 
            raise FileExistsError(
                f"Experiment '{experiment_name}' already exists."
            )

    model_dir = experiment_dir / "model"
    metrics_dir = experiment_dir / "metrics"
    figures_dir = experiment_dir / "figures"
    reports_dir = experiment_dir / "reports"

    for directory in (
        model_dir,
        metrics_dir,
        figures_dir,
        reports_dir
    ):
        directory.mkdir(
            parents= True,
            exist_ok=True,
        )

    joblib.dump(
        artifacts.model,
        model_dir / "model.joblib",
    )

    joblib.dump(
        artifacts.preprocessor,
        model_dir / "preprocessor.joblib"
    )

    with open(
        model_dir / "feature_names.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            artifacts.feature_names,
            f,
            indent=4,
        )

    shutil.copy2(
        CONFIG_DIR / settings.files.config_file,
        model_dir / settings.files.config_file
    )

    logger.info(f"Artifacts saved to '{experiment_dir}'.")

    return experiment_dir