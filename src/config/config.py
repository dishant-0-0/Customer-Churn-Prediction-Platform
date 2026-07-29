from pathlib import Path
import yaml
from src.config.paths import CONFIG_DIR

CONFIG_PATH = CONFIG_DIR / "config.yaml"

with open(CONFIG_PATH, "r") as file:
    settings = yaml.safe_load(file)