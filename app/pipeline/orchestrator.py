from __future__ import annotations

from pathlib import Path

from app.core.config import AppConfig
from app.core.hardware import detect_hardware, resolve_backend
from app.core.models import PipelineProgress, PipelineResult, PipelineStage
from app.pipeline.align import align_segments
from app.pipeline.diarize import diarize
from app.pipeline.export import export_all
from app.pipeline.inspect import inspect_media
from app.pipeline.preprocess import preprocess_media
from app.pipeline.transcribe import transcribe
from app.storage.recent_jobs import append_recent_job


def _emit(callback, stage: PipelineStage, message: str, percent: float | None = None) -> None:
    if callback:
        callback(PipelineProgress(stage=stage, message=message, percent=percent))


def _resolve_output_dir(media_path: Path, config: AppConfig) -> Path:
    configured_output_dir = config.general.output_dir.strip()
    output_dir = (
        Path(configured_output_dir).expanduser()
        if configured_output_dir
        else media_path.parent / "outputs"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_job(media_path: Path, config: AppConfig, progress_cb=None) -> PipelineResult:
    _emit(progress_cb, PipelineStage.INSPECT, "Inspecting media", 5)
    media = inspect_media(media_path, config)

    _emit(progress_cb, PipelineStage.HARDWARE, "Detecting hardware", 10)
    hardware = detect_hardware()
    backend = resolve_backend(config, hardware)

    job_dir = _resolve_output_dir(media_path, config)

    _emit(progress_cb, PipelineStage.PREPROCESS, "Normalizing media to WAV", 20)
    audio_path = preprocess_media(media, job_dir, config)

    _emit(progress_cb, PipelineStage.TRANSCRIBE, "Running faster-whisper", 50)
    transcript_segments = transcribe(audio_path, config, backend)

    speaker_turns = []
    if config.diarization.enabled:
        _emit(progress_cb, PipelineStage.DIARIZE, "Running pyannote diarization", 75)
        speaker_turns = diarize(audio_path, config, backend)

    _emit(progress_cb, PipelineStage.ALIGN, "Aligning transcript with speaker turns", 85)
    aligned_segments = align_segments(transcript_segments, speaker_turns)

    result = PipelineResult(
        input_media=media,
        backend=backend,
        transcript_segments=transcript_segments,
        speaker_turns=speaker_turns,
        aligned_segments=aligned_segments,
        artifacts=[],
        output_dir=job_dir,
    )

    _emit(progress_cb, PipelineStage.EXPORT, "Exporting artifacts", 95)
    result.artifacts = export_all(result, config.export.formats)

    append_recent_job(result)
    _emit(progress_cb, PipelineStage.DONE, "Finished", 100)
    return result
