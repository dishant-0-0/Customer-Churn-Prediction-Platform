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
    X_train : np.ndarray
        Processed training features.
    
    X_test : np.ndarray
        Processed testing features.
    
    y_train: pd.Series
        Training target.

    y_test: pd.Series
        Testing target.

    preprocessor: ColumnTransformer
        Fitted preprocessing pipeline.
    """

    X_train: np.ndarray
    X_test: np.ndarray

    y_train: pd.Series
    y_test: pd.Series

    preprocessor: ColumnTransformer

    numerical_columns: list[str]
    categorical_columns: list[str]