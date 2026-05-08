from __future__ import annotations

import tomllib
from pathlib import Path

from app.core.models import AppConfigModel
from app.core.paths import app_config_dir


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _to_toml(model: AppConfigModel) -> str:
    lines: list[str] = []
    for section_name, section_data in model.model_dump().items():
        lines.append(f"[{section_name}]")
        for key, value in section_data.items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


class AppConfig(AppConfigModel):
    @classmethod
    def load(cls) -> "AppConfig":
        path = config_path()
        if not path.exists():
            return cls()
        data = tomllib.loads(path.read_text())
        return cls.model_validate(data)

    def save(self) -> Path:
        path = config_path()
        path.write_text(_to_toml(self))
        return path


def config_path() -> Path:
    return app_config_dir() / "config.toml"


def ensure_config_file() -> Path:
    path = config_path()
    if not path.exists():
        AppConfig().save()
    return path


def load_config() -> AppConfig:
    return AppConfig.load()
