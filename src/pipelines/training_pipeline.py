"""
Training data preparation pipeline.
"""

from __future__ import annotations

import pandas as pd

from src.config.config import settings
from src.core import InferenceArtifacts, ProcessedData, TrainingResult
from src.data.loader import load_data
from src.data.preprocessing import (
    build_preprocessor,
    get_feature_types,
    split_data,
)
from src.explainability.shap import generate_explanation
from src.features.feature_engineering import feature_engineering_pipeline
from src.models.evaluate import evaluate_model
from src.models.registry import get_model
from src.models.train import train_model
from src.persistence.save import save_artifacts, save_figures, save_metrics
from src.reporting import generate_html_report
from src.tracking import log_experiment
from src.utils.feature_names import clean_feature_names
from src.utils.logger import get_logger
from src.visualization import generate_visualization

logger = get_logger(__name__)


def prepare_training_data() -> ProcessedData:
    """
    Prepare dataset for model training.
    """

    logger.info("Starting training data preparation pipeline.")

    logger.info("Loading processed dataset.")
    df = load_data(stage="processed", filename=settings.files.processed_file)
    logger.info("Loaded processed dataset with shape: %s", df.shape)

    logger.info("Applying feature engineering.")
    high_value_threshold = df["CLTV"].median()
    df = feature_engineering_pipeline(df, high_value_threshold)
    logger.info("Feature engineering completed. Dataset shape: %s", df.shape)

    logger.info("Splitting data.")
    X_train, X_test, y_train, y_test = split_data(
        df=df, target=settings.data.target
    )
    logger.info(
        "Data split completed. Training samples: %d, Testing samples: %d",
        len(X_train),
        len(X_test),
    )

    logger.info("Identifying feature types.")
    numerical_cols, categorical_cols = get_feature_types(X_train)
    logger.info(
        "Identified %d numerical features and %d categorical features.",
        len(numerical_cols),
        len(categorical_cols),
    )

    logger.info("Building preprocessing pipeline.")
    preprocessor = build_preprocessor(
        numerical_cols=numerical_cols, categorical_cols=categorical_cols
    )
    logger.info("Preprocessing pipeline built successfully.")

    logger.info("Fitting preprocessing pipeline.")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out().tolist()
    feature_names = clean_feature_names(feature_names)

    logger.info("Preprocessing completed.")

    logger.info("Training data preparation completed successully.")

    return ProcessedData(
        X_train=X_train,
        X_test=X_test,
        X_train_processed=X_train_processed,
        X_test_processed=X_test_processed,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        feature_names=feature_names,
        numerical_columns=numerical_cols,
        categorical_columns=categorical_cols,
        high_value_threshold=high_value_threshold,
    )


def run_training_pipeline(force: bool = False) -> TrainingResult:
    """
    Run training pipeline and save artifacts.
    """

    processed = prepare_training_data()

    logger.info("Creating model instance.")

    model = get_model()

    logger.info(f"Training '{settings.training.model.name}' model.")

    model = train_model(processed, model)

    logger.info(f"Evaluating '{settings.training.model.name}' model.")

    evaluation = evaluate_model(processed, model)

    artifacts = InferenceArtifacts(
        model=model,
        preprocessor=processed.preprocessor,
        feature_names=processed.feature_names,
        high_value_threshold=processed.high_value_threshold,
    )

    logger.info("Saving artifacts.")

    artifacts_path = save_artifacts(artifacts, force=force)

    logger.info("Saving metrics.")

    save_metrics(
        evaluation,
    )

    X_shap = pd.DataFrame(
        processed.X_test_processed[:250],
        columns=processed.feature_names,
    )

    logger.info("Generating explanation.")

    explanation = generate_explanation(
        model=model,
        X=X_shap,
    )

    logger.info("Generating visualization.")

    figures = generate_visualization(
        evaluation=evaluation,
        explainability=explanation,
        model=model,
        feature_names=processed.feature_names,
    )

    logger.info("Saving visualization.")

    save_figures(figures)

    training_result = TrainingResult(
        model=model,
        evaluation=evaluation,
        artifacts=artifacts,
        artifacts_path=artifacts_path,
    )

    logger.info("Generating Report.")

    generate_html_report(training_result)

    logger.info("Logging Experiment.")

    log_experiment(training_result=training_result)

    return training_result
