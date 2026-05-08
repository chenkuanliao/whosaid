from __future__ import annotations

from pathlib import Path

from app.core.config import AppConfig
from app.pipeline.orchestrator import _resolve_output_dir


def test_resolve_output_dir_defaults_to_outputs_next_to_media(tmp_path: Path) -> None:
    media_path = tmp_path / "audio.wav"
    media_path.write_bytes(b"")

    output_dir = _resolve_output_dir(media_path, AppConfig())

    assert output_dir == tmp_path / "outputs"
    assert output_dir.is_dir()


def test_resolve_output_dir_reuses_existing_default_outputs_dir(tmp_path: Path) -> None:
    media_path = tmp_path / "audio.wav"
    media_path.write_bytes(b"")
    existing_output_dir = tmp_path / "outputs"
    existing_output_dir.mkdir()
    marker = existing_output_dir / ".marker"
    marker.write_text("keep")

    output_dir = _resolve_output_dir(media_path, AppConfig())

    assert output_dir == existing_output_dir
    assert marker.read_text() == "keep"


def test_resolve_output_dir_uses_configured_output_dir(tmp_path: Path) -> None:
    media_path = tmp_path / "audio.wav"
    media_path.write_bytes(b"")
    configured_output_dir = tmp_path / "configured"
    config = AppConfig()
    config.general.output_dir = str(configured_output_dir)

    output_dir = _resolve_output_dir(media_path, config)

    assert output_dir == configured_output_dir
    assert output_dir.is_dir()
