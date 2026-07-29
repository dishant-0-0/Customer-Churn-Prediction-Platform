from pydantic import BaseModel

class ProjectConfig(BaseModel):
    name: str
    version: str

class PathConfig(BaseModel):
    config_file: str
    raw_file: str
    interim_file: str
    processed_file: str

class DataConfig(BaseModel):
    target: str
    test_size: float
    validation_size: float
    random_state: int

class FeatureEngineeringConfig(BaseModel):
    high_value_threshold: float

class TrainingConfig(BaseModel):
    model: str

class LoggingConfig(BaseModel):
    level: str