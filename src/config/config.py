from pathlib import Path
import yaml
from src.config.paths import CONFIG_DIR
from src.config.settings import Settings

CONFIG_PATH = CONFIG_DIR / "config.yaml"

def load_settings() -> Settings:
    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    return Settings(**raw)

settings = load_settings()