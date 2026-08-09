"""
Deployment artifacts.
"""

from __future__ import annotations
from dataclasses import dataclass
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer


@dataclass(frozen=True, slots=True)
class InferenceArtifacts:
    """
    Runtime artifacts required for inference.

    Attributes
    ----------
    model: BaseEstimator
        Trained model.
    
    preprocessor: ColumnTransformer
        Fitted preprocessing pipeline.

    feature_names: list[str]
        List of feature names.
    """

    model: BaseEstimator

    preprocessor: ColumnTransformer

    feature_names: list[str]