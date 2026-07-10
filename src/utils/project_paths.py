from pathlib import Path
import sys


def get_project_root(start: Path | str | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()

    for candidate in [current, *current.parents]:
        if (candidate / "configs").exists() and (candidate / "src").exists():
            return candidate

    return current


def get_config_path(start: Path | str | None = None) -> Path:
    return get_project_root(start) / "configs" / "config.yaml"


def ensure_project_root_on_path(start: Path | str | None = None) -> Path:
    project_root = get_project_root(start)
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return project_root
