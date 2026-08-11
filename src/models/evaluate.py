"""
Model evaludation utilities.
"""

from __future__ import annotations
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
from src.config.config import settings
from src.core import ProcessedData, EvaluationResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

def evaluate_model(
    processed: ProcessedData,
    model: BaseEstimator,
) -> EvaluationResult:
    """
    Evaluate a trained classification model.
    """

    logger.info("Evaluating model: %s", model.__class__.__name__)

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            f"{model.__class__.__name__} does not implement 'predict_proba()'."
        )

    y_prob = model.predict_proba(processed.X_test_processed)[:, 1]

    y_pred = (y_prob >= settings.evaluation.threshold).astype(int)

    accuracy = accuracy_score(processed.y_test, y_pred)

    precision = precision_score(processed.y_test, y_pred)

    recall = recall_score(processed.y_test, y_pred)

    f1 = f1_score(processed.y_test, y_pred)

    roc_auc = roc_auc_score(processed.y_test, y_prob)

    cm = confusion_matrix(processed.y_test, y_pred)

    fpr, tpr, thresholds = roc_curve(processed.y_test, y_prob)

    logger.info("Evaluation completed successfully.")

    return EvaluationResult(
        y_pred= y_pred,
        y_prob=y_prob,
        accuracy= accuracy,
        precision = precision,
        recall= recall,
        f1= f1,
        roc_auc= roc_auc,
        confusion_matrix= cm,
        fpr=fpr,
        tpr=tpr,
        roc_thresholds= thresholds
    )