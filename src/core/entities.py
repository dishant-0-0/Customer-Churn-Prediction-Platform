"""
Data models used throughout the ML pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator
from src.persistence import InferenceArtifacts

@dataclass(frozen=True, slots=True)
class ProcessedData:
    """
    Container for processed training and testing datasets.

    Attributes
    ----------
    X_train : pd.DataFrame
        Training features before preprocessing.
    
    X_test : pd.DataFrame
        Testing features before preprocessing.

    X_train_processed : np.ndarray
        Processed training features.
    
    X_test_processed : np.ndarray
        Processed testing features.
    
    y_train: pd.Series
        Training target.

    y_test: pd.Series
        Testing target.

    preprocessor: ColumnTransformer
        Fitted preprocessing pipeline.

    feature_names: list[str]
        List of feature names after preprocessing.

    numerical_columns: list[str]
        List of numerical feature names.

    categorical_columns: list[str]
        List of categorical feature names.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame

    X_train_processed: np.ndarray
    X_test_processed: np.ndarray

    y_train: pd.Series
    y_test: pd.Series

    preprocessor: ColumnTransformer

    feature_names: list[str]

    numerical_columns: list[str]
    categorical_columns: list[str]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """
    Container for model evaluation results.
    
    Attributes
    ----------
    y_pred : np.ndarray
        Predicted class labels.
    
    y_prob: np.ndarray
        Predicted probabilities for the positive class.

    accuracy: float
        Accuracy score.

    precision: float
        Precision score.

    recall: float
        Recall score.

    f1: float
        F1 score.

    roc_auc: float
        ROC-AUC score.

    confusion_matrix: np.ndarray
        Confusion matrix.

    fpr: np.ndarray
        False positive rates for ROC curve.

    tpr: np.ndarray
        True positive rates for ROC curve.

    thresholds: np.ndarray
        Decision thresholds for ROC curve.
    """

    y_pred: np.ndarray
    y_prob: np.ndarray

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    confusion_matrix: np.ndarray

    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray


@dataclass(frozen=True, slots= True)
class TrainingResult:
    """
    Output of the complete training workflow.
    """

    model: BaseEstimator
    evaluation: EvaluationResult
    artifacts: InferenceArtifacts
    artifacts_path: Path