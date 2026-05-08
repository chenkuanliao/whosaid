from __future__ import annotations

from pathlib import Path

from platformdirs import PlatformDirs

APP_DIRS = PlatformDirs("whosaid", "whosaid")


def app_config_dir() -> Path:
    path = Path(APP_DIRS.user_config_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def app_data_dir() -> Path:
    path = Path(APP_DIRS.user_data_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def app_log_dir() -> Path:
    path = app_data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def app_output_dir() -> Path:
    path = app_data_dir() / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path
