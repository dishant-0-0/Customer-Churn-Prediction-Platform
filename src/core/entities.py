"""
Data models used throughout the ML pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

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