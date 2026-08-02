from pydantic import BaseModel, ConfigDict
from typing import Any

class AppConfig(BaseModel):
    model_config = ConfigDict(
        frozen= True,
        extra= "forbid"
    )

class ProjectConfig(AppConfig):
    name: str
    version: str

class FilesConfig(AppConfig):
    raw_file: str
    interim_file: str
    processed_file: str

class DataConfig(AppConfig):
    target: str
    test_size: float
    validation_size: float
    random_state: int

class FeatureEngineeringConfig(AppConfig):
    required_columns: list[str]
    drop_columns: list[str]
    drop_location_columns: list[str]
    tenure_bins: list[int]
    tenure_labels: list[str]
    high_value_strategy: str

class ModelConfig(AppConfig):
    name: str
    params: dict[str, Any] = {}

class TrainingConfig(AppConfig):
    model: ModelConfig

class LoggingConfig(AppConfig):
    level: str
    file: str
    console: bool

class Settings(AppConfig):
    project: ProjectConfig
    files: FilesConfig
    data: DataConfig
    feature_engineering: FeatureEngineeringConfig
    training: TrainingConfig
    logging: LoggingConfig