"""
Shared FastAPI dependencies.
"""

from __future__ import annotations
from src.core import InferenceArtifacts
from src.persistence.load import load_artifacts
from src.utils.logger import get_logger

logger = get_logger(__name__)

_artifacts: InferenceArtifacts | None = None


def load_inference_artifacts() -> None:
    """
    Load inference artifacts into memory.
    """

    global _artifacts

    if _artifacts is not None:
        logger.info("Inference artifacts already loaded.")
        return

    logger.info("Loading inference artifacts.")

    _artifacts = load_artifacts()

    logger.info("Inference artifacts loaded successfully.")


def get_inference_artifacts() -> InferenceArtifacts:
    """
    Return the loaded inference artifacts.
    """

    if _artifacts is None:
        raise RuntimeError(
            "Inference artifacts have not been initialized."
        )

    return _artifacts