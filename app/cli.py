from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from app.core.config import AppConfig, config_path, ensure_config_file, load_config
from app.core.models import PipelineProgress
from app.pipeline.inspect import inspect_media
from app.pipeline.orchestrator import run_job

cli = typer.Typer(no_args_is_help=True, rich_markup_mode="markdown")
config_app = typer.Typer(no_args_is_help=True)
cli.add_typer(config_app, name="config")
console = Console()


def _progress_printer(progress: PipelineProgress) -> None:
    message = f"[{progress.stage.value}] {progress.message}"
    if progress.percent is not None:
        message += f" ({progress.percent:.0f}%)"
    console.print(message)


@cli.command()
def doctor() -> None:
    from app.core.hardware import detect_hardware
    from app.services.ffmpeg_service import check_ffmpeg
    from app.services.pyannote_service import pyannote_readiness
    from app.services.whisper_service import whisper_readiness

    hardware = detect_hardware()
    ffmpeg = check_ffmpeg()
    whisper = whisper_readiness()
    pyannote = pyannote_readiness()

    table = Table(title="WhoSaid Doctor")
    table.add_column("Check")
    table.add_column("Result")
    table.add_row("Backend", hardware.summary())
    table.add_row("FFmpeg", "ok" if ffmpeg.available else ffmpeg.message)
    table.add_row("faster-whisper", whisper)
    table.add_row("pyannote", pyannote)
    table.add_row(
        "Hugging Face token",
        "present" if pyannote.startswith("ready") else "missing or gated",
    )
    console.print(table)


@cli.command()
def inspect(media_path: Path) -> None:
    config = load_config()
    media = inspect_media(media_path, config)
    console.print_json(media.model_dump_json(indent=2))


@cli.command()
def run(
    media_path: Path,
    model: str | None = None,
    language: str | None = None,
    diarization: Annotated[bool, typer.Option("--diarization/--no-diarization")] = True,
    exports: Annotated[list[str] | None, typer.Option("--exports")] = None,
) -> None:
    config = load_config()
    if model:
        config.transcription.model = model
    if language:
        config.transcription.language = language
    config.diarization.enabled = diarization
    if exports:
        config.export.formats = exports

    result = run_job(media_path, config, _progress_printer)
    console.print_json(result.model_dump_json(indent=2))

@config_app.command("init")
def config_init() -> None:
    path = ensure_config_file()
    console.print(f"Config initialized at {path}")


@config_app.command("show")
def config_show() -> None:
    ensure_config_file()
    path = config_path()
    console.print(f"Config path: {path}")
    config = AppConfig.load()
    console.print_json(config.model_dump_json(indent=2))
