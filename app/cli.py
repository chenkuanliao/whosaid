from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from app.core.config import AppConfig, config_path, ensure_config_file, load_config
from app.core.models import PipelineProgress, PipelineResult
from app.pipeline.inspect import inspect_media
from app.pipeline.orchestrator import run_job
from app.pipeline.speaker_linking import (
    apply_speaker_names,
    exported_formats,
    find_result_json,
    load_result,
    rewrite_exports,
    speaker_ids,
)
from app.storage.recent_jobs import load_recent_jobs

cli = typer.Typer(no_args_is_help=True, rich_markup_mode="markdown")
config_app = typer.Typer(no_args_is_help=True)
cli.add_typer(config_app, name="config")
console = Console()


def _progress_printer(progress: PipelineProgress) -> None:
    message = f"[{progress.stage.value}] {progress.message}"
    if progress.percent is not None:
        message += f" ({progress.percent:.0f}%)"
    console.print(message)


def _print_run_summary(result: PipelineResult) -> None:
    table = Table(title="WhoSaid Run Summary")
    table.add_column("Field")
    table.add_column("Value", overflow="ignore", no_wrap=True)
    table.add_row("Output directory", str(result.output_dir))
    table.add_row("Input media", str(result.input_media.source_path))
    table.add_row("Media type", result.input_media.media_type.value)
    table.add_row("Duration (seconds)", f"{result.input_media.duration_seconds:.2f}")
    table.add_row("Backend", result.backend.mode)
    table.add_row("Compute type", result.backend.compute_type)
    table.add_row("Transcript segments", str(len(result.transcript_segments)))
    table.add_row("Speaker turns", str(len(result.speaker_turns)))
    table.add_row("Aligned segments", str(len(result.aligned_segments)))
    artifacts = (
        ", ".join(f"{artifact.format}: {artifact.path.name}" for artifact in result.artifacts)
        or "none"
    )
    table.add_row("Artifacts", artifacts)
    console.print(table)


@cli.command()
def doctor() -> None:
    from app.core.hardware import detect_hardware
    from app.services.ffmpeg_service import check_ffmpeg
    from app.services.pyannote_service import pyannote_readiness
    from app.services.whisper_service import whisper_readiness

    config = load_config()
    hardware = detect_hardware()
    ffmpeg = check_ffmpeg()
    whisper = whisper_readiness()
    pyannote = pyannote_readiness(config.diarization.model)

    table = Table(title="WhoSaid Doctor")
    table.add_column("Check")
    table.add_column("Result")
    table.add_row("Backend", hardware.summary())
    table.add_row("FFmpeg", "ok" if ffmpeg.available else ffmpeg.message)
    table.add_row("faster-whisper", whisper)
    table.add_row("pyannote", pyannote)
    table.add_row(
        "Hugging Face token",
        "present" if os.getenv("HUGGINGFACE_HUB_TOKEN") else "missing",
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
    _print_run_summary(result)


@cli.command("link-speakers")
def link_speakers(
    output_path: Annotated[
        Path | None,
        typer.Argument(
            help="Output directory or .transcript.json file. Defaults to the latest run."
        ),
    ] = None,
) -> None:
    if output_path is None:
        recent_jobs = load_recent_jobs()
        if not recent_jobs:
            console.print("No previous run output found. Run `whosaid run` first.")
            raise typer.Exit(1)
        output_path = recent_jobs[0].output_dir

    result_path = find_result_json(output_path)
    if result_path is None:
        console.print(f"No transcript JSON output found at {output_path}.")
        raise typer.Exit(1)

    try:
        result = load_result(result_path)
    except Exception as exc:
        console.print(f"Could not read transcript output at {result_path}: {exc}")
        raise typer.Exit(1) from exc

    speakers = speaker_ids(result)
    if not speakers:
        console.print(f"No speaker ids found in {result_path}. Nothing to link.")
        raise typer.Exit(1)

    names: dict[str, str] = {}
    console.print("Link speaker ids to names. Press Enter to skip a speaker.")
    for speaker in speakers:
        name = typer.prompt(f"Name for {speaker}", default="", show_default=False)
        if name.strip():
            names[speaker] = name.strip()

    if not names:
        console.print("No speaker names provided. Outputs were left unchanged.")
        return

    apply_speaker_names(result, names)
    formats = exported_formats(result, result_path)
    written = rewrite_exports(result, result_path, formats)

    console.print("Updated speaker names in:")
    for path in written:
        console.print(f"- {path}")


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
