"""
Tests for persistence load utilities.
"""

from __future__ import annotations
import json
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from src.persistence import load


def test_load_json(
    tmp_path,
):
    """
    JSON file should be loaded successfully.
    """

    json_file = tmp_path / "test.json"

    expected = {
        "a": 1,
        "b": 2,
    }

    with json_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            expected,
            file,
        )

    result = load._load_json(
        json_file,
    )

    assert result == expected


def test_load_artifacts_missing_experiment(
    patched_artifact_dirs,
):
    """
    Missing experiment directory should raise FileNotFoundError.
    """

    with pytest.raises(FileNotFoundError):
        load.load_artifacts(
            experiment_name="missing_experiment",
        )


def test_load_artifacts_missing_files(
    patched_artifact_dirs,
):
    """
    Missing artifact files should raise FileNotFoundError.
    """

    experiment_dir = (
        patched_artifact_dirs
        / "missing_files"
    )

    model_dir = experiment_dir / "model"

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pytest.raises(FileNotFoundError):
        load.load_artifacts(
            experiment_name="missing_files",
        )


def test_load_artifacts(
    create_saved_artifacts,
):
    """
    Artifacts should be loaded successfully.
    """

    experiment_name, experiment_dir = (
        create_saved_artifacts(
            experiment_name="test_experiment",
        )
    )

    artifacts = load.load_artifacts(
        experiment_name=experiment_name,
    )

    assert experiment_dir.exists()

    assert isinstance(
        artifacts.model,
        DummyClassifier,
    )

    assert isinstance(
        artifacts.preprocessor,
        StandardScaler,
    )

    assert artifacts.feature_names == [
        "A",
        "B",
    ]

    assert (
        artifacts.high_value_threshold
        == 5000.0
    )


def test_load_artifacts_default_experiment_name(
    create_saved_artifacts,
):
    """
    Default experiment name should be used.
    """

    experiment_name = (
        f"{load.settings.training.model.name}_"
        f"{load.settings.artifacts.version}"
    )

    create_saved_artifacts(
        experiment_name=experiment_name,
        feature_names=[
            "Monthly Charges",
        ],
        high_value_threshold=6000.0,
    )

    artifacts = load.load_artifacts()

    assert artifacts.feature_names == [
        "Monthly Charges",
    ]

    assert (
        artifacts.high_value_threshold
        == 6000.0
    )


def test_loaded_model_is_usable(
    create_saved_artifacts,
):
    """
    Loaded model and preprocessor should be usable.
    """

    experiment_name, _ = (
        create_saved_artifacts(
            experiment_name="usable",
        )
    )

    artifacts = load.load_artifacts(
        experiment_name=experiment_name,
    )

    transformed = (
        artifacts.preprocessor.transform(
            [[0.5]]
        )
    )

    prediction = artifacts.model.predict(
        transformed,
    )

    assert prediction.shape == (1,)