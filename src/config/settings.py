from typing import Any

from pydantic import BaseModel, ConfigDict


class AppConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ProjectConfig(AppConfig):
    name: str
    version: str


class FilesConfig(AppConfig):
    raw_file: str
    interim_file: str
    processed_file: str
    config_file: str


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


class EvaluationConfig(AppConfig):
    threshold: float


class LoggingConfig(AppConfig):
    level: str
    file: str
    console: bool


class ArtifactsConfig(AppConfig):
    version: str


class FigureSizeConfig(AppConfig):
    width: int
    height: int


class FontConfig(AppConfig):
    title_size: int
    label_size: int
    tick_size: int


class ConfusionMatrixConfig(AppConfig):
    cmap: str
    annot: bool


class RocCurveConfig(AppConfig):
    color: str
    linewidth: int


class FeatureImportanceConfig(AppConfig):
    top_n: int
    color: str


class ShapConfig(AppConfig):
    max_display: int


class VisualizationConfig(AppConfig):
    style: str
    dpi: int
    figsize: FigureSizeConfig
    font: FontConfig
    confusion_matrix: ConfusionMatrixConfig
    roc_curve: RocCurveConfig
    feature_importance: FeatureImportanceConfig
    shap: ShapConfig


class TrackingConfig(AppConfig):
    enabled: bool
    experiment_name: str
    tracking_uri: str


class CorsConfig(AppConfig):
    allow_origins: list[str]
    allow_credentials: bool
    allow_methods: list[str]
    allow_headers: list[str]


class ApiConfig(AppConfig):
    title: str
    description: str
    version: str
    cors: CorsConfig


class Settings(AppConfig):
    project: ProjectConfig
    files: FilesConfig
    data: DataConfig
    feature_engineering: FeatureEngineeringConfig
    training: TrainingConfig
    logging: LoggingConfig
    evaluation: EvaluationConfig
    artifacts: ArtifactsConfig
    visualization: VisualizationConfig
    tracking: TrackingConfig
    api: ApiConfig
