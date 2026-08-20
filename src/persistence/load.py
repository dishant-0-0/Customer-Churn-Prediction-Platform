"""
Utilities for loading deployment artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import joblib

from src.config.config import settings
from src.config.paths import ARTIFACTS_DIR
from src.core import InferenceArtifacts
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _load_json(
    input_path: Path,
) -> dict[str, Any] | list[Any]:
    """
    Load JSON data from disk.
    """

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return cast(
            dict[str, Any] | list[Any],
            json.load(file),
        )


def load_artifacts(
    experiment_name: str | None = None,
) -> InferenceArtifacts:
    """
    Load deployment artifacts from disk.
    """

    if experiment_name is None:
        experiment_name = (
            f"{settings.training.model.name}_{settings.artifacts.version}"
        )

    experiment_dir = ARTIFACTS_DIR / experiment_name
    model_dir = experiment_dir / "model"

    if not experiment_dir.exists():
        raise FileNotFoundError(
            f"Experiment directory not found: {experiment_dir}"
        )

    required_files = {
        "model": model_dir / "model.joblib",
        "preprocessor": model_dir / "preprocessor.joblib",
        "feature_names": model_dir / "feature_names.json",
        "metadata": model_dir / "metadata.json",
    }

    for artifact_name, artifact_path in required_files.items():
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Required artifact '{artifact_name}' "
                f"not found: {artifact_path}"
            )

    logger.info(f"Loading artifacts from '{experiment_dir}'.")

    model = joblib.load(required_files["model"])

    preprocessor = joblib.load(required_files["preprocessor"])

    metadata = cast(
        dict[str, Any],
        _load_json(required_files["metadata"]),
    )

    feature_names = cast(
        list[str], _load_json(required_files["feature_names"])
    )

    logger.info(f"Artifacts loaded successfully from '{experiment_dir}'.")

    return InferenceArtifacts(
        model=model,
        preprocessor=preprocessor,
        feature_names=feature_names,
        high_value_threshold=metadata["high_value_threshold"],
    )
