"""
Tests for persistence save utilities.
"""

from __future__ import annotations
import matplotlib

matplotlib.use("Agg")

import json
import matplotlib.pyplot as plt
import pytest
from src.persistence import save



def test_get_experiment_dir(
    tmp_path,
    monkeypatch,
):
    """
    Experiment directory should be created correctly.
    """

    monkeypatch.setattr(
        save,
        "ARTIFACTS_DIR",
        tmp_path,
    )

    experiment_dir = save.get_experiment_dir()

    assert experiment_dir.parent == tmp_path


def test_save_artifacts(
    mock_inference_artifacts,
    patched_artifact_dirs,
):
    """
    Artifacts should be saved successfully.
    """

    experiment_dir = save.save_artifacts(
        mock_inference_artifacts,
    )

    model_dir = experiment_dir / "model"

    assert model_dir.exists()

    assert (model_dir / "model.joblib").exists()
    assert (model_dir / "preprocessor.joblib").exists()
    assert (model_dir / "feature_names.json").exists()
    assert (model_dir / "metadata.json").exists()

    with (model_dir / "feature_names.json").open(
        encoding="utf-8",
    ) as file:
        feature_names = json.load(file)

    assert (
        feature_names
        == mock_inference_artifacts.feature_names
    )

    with (model_dir / "metadata.json").open(
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    assert (
        metadata["high_value_threshold"]
        == mock_inference_artifacts.high_value_threshold
    )


def test_save_artifacts_existing_directory(
    mock_inference_artifacts,
    patched_artifact_dirs,
):
    """
    Saving twice without force should raise FileExistsError.
    """

    save.save_artifacts(
        mock_inference_artifacts,
    )

    with pytest.raises(FileExistsError):
        save.save_artifacts(
            mock_inference_artifacts,
        )


def test_save_metrics(
    mock_evaluation_result,
    tmp_path,
    monkeypatch,
):
    """
    Metrics should be written to JSON.
    """

    monkeypatch.setattr(
        save,
        "get_experiment_paths",
        lambda: {
            "metrics": tmp_path,
        },
    )

    output = save.save_metrics(
        mock_evaluation_result,
    )

    assert output.exists()

    with output.open(
        encoding="utf-8"
    ) as file:
        metrics = json.load(file)

    assert (
        metrics["accuracy"]
        == mock_evaluation_result.accuracy
    )
    assert (
        metrics["precision"]
        == mock_evaluation_result.precision
    )
    assert (
        metrics["recall"]
        == mock_evaluation_result.recall
    )
    assert (
        metrics["f1"]
        == mock_evaluation_result.f1
    )
    assert (
        metrics["roc_auc"]
        == mock_evaluation_result.roc_auc
    )
    assert (
        metrics["average_precision"]
        == mock_evaluation_result.average_precision
    )


def test_save_figure(
    tmp_path,
):
    """
    Figure should be saved successfully.
    """

    fig, ax = plt.subplots()

    ax.plot(
        [1, 2],
        [3, 4],
    )

    output = tmp_path / "figure.png"

    returned = save.save_figure(
        fig,
        output,
    )

    assert returned == output
    assert output.exists()
    assert output.suffix == ".png"


def test_save_figures(
    tmp_path,
    monkeypatch,
):
    """
    Multiple figures should be saved.
    """

    monkeypatch.setattr(
        save,
        "get_experiment_paths",
        lambda: {
            "figures": tmp_path,
        },
    )

    fig1, _ = plt.subplots()
    fig2, _ = plt.subplots()

    saved = save.save_figures(
        {
            "roc": fig1,
            "cm": fig2,
        }
    )

    assert saved == [
        tmp_path / "roc.png",
        tmp_path / "cm.png",
    ]
    assert (tmp_path / "roc.png").exists()
    assert (tmp_path / "cm.png").exists()


def test_save_artifacts_force_overwrite(
    mock_inference_artifacts,
    patched_artifact_dirs,
):
    """
    Existing experiment should be overwritten when force=True.
    """

    save.save_artifacts(
        mock_inference_artifacts,
    )

    experiment_dir = save.save_artifacts(
        mock_inference_artifacts,
        force=True,
    )

    model_dir = experiment_dir / "model"

    assert experiment_dir.exists()

    assert (model_dir / "model.joblib").exists()
    assert (model_dir / "preprocessor.joblib").exists()